import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)


def create_gaussian_kernel(kernel_size, sigma):
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    kernel = kernel / np.sum(kernel)
    return torch.tensor(kernel, dtype=torch.float32)


# 定义高斯低通滤波器层
class GaussianBlurLayer(nn.Module):
    def __init__(self, kernel_size=15, sigma=2.2):
        super(GaussianBlurLayer, self).__init__()
        kernel = create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer('weight', kernel.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # 扩展 kernel 以匹配输入的通道数
        kernel = self.weight.expand(x.size(1), 1, self.weight.size(2), self.weight.size(3))
        return F.conv2d(x, kernel, padding=self.weight.size(2) // 2, groups=x.size(1))


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, modes_x, modes_y, fourier_weight=None, mode='full'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.mode = mode
        self.fourier_weight = fourier_weight
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_x, modes_y]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        B, I, M, N = x.shape

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        if self.mode == 'full':
            out_ft[:, :, :self.modes_x, :] = torch.einsum(
                "bixy,iox->boxy",
                x_ftx[:, :, :self.modes_x, :],
                torch.view_as_complex(self.fourier_weight[0]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :self.modes_x, :] = x_ftx[:, :, :self.modes_x, :]

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        if self.mode == 'full':
            out_ft[:, :, :, :self.modes_y] = torch.einsum(
                "bixy,ioy->boxy",
                x_fty[:, :, :, :self.modes_y],
                torch.view_as_complex(self.fourier_weight[1]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :, :self.modes_y] = x_fty[:, :, :, :self.modes_y]

        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy

        return x


class Residual(nn.Module):
    def __init__(self, in_width,out_width):
        super(Residual, self).__init__()
        self.conv0 = nn.Conv2d(in_width, out_width, 1)
        self.conv1 = nn.Conv2d(out_width, out_width, 1)
        self.bn0 = nn.BatchNorm2d(out_width)
        self.bn1 = nn.BatchNorm2d(out_width)

    def forward(self, x):
        y = self.conv0(x)
        y = F.gelu(y)
        y = self.bn0(y)
        y = self.conv1(y)
        y = self.bn1(y)
        y += x
        return y


class MLP(nn.Module):
    def __init__(self, in_width, out_width, mid_width):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_width, mid_width, 1)
        self.mlp2 = nn.Conv2d(mid_width, out_width, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, polling=True, bn=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU(),
        ]
        if dropout:
            layers.append(nn.Dropout(0.25))
        self.encode = nn.Sequential(*layers)
        self.pool = None
        if polling:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bn=False, dropout=False):
        super(_DecoderBlock, self).__init__()
        if dropout:
            self.decode = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
                nn.GELU(),
                nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
            )
        else:
            self.decode = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
                nn.GELU(),
                nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
                nn.GELU(),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
            )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(in_channels, 32, polling=False, bn=bn)
        self.enc2 = _EncoderBlock(32, 64, bn=bn)
        self.enc3 = _EncoderBlock(64, 128, bn=bn)
        self.enc4 = _EncoderBlock(128, 256, bn=bn, dropout=False)
        self.polling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlock(256, 512, 256, bn=bn)
        self.dec4 = _DecoderBlock(512, 256, 128, bn=bn)
        self.dec3 = _DecoderBlock(256, 128, 64, bn=bn)
        self.dec2 = _DecoderBlock(128, 64, 32, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if bn else nn.GroupNorm(32, 64),
            nn.GELU()
        )
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)#(20,32,221,51)
        enc2 = self.enc2(enc1)#(20,64,221,51)
        enc3 = self.enc3(enc2)#(20,128,221,51)
        enc4 = self.enc4(enc3)#(20,256,221,51)
        center = self.center(self.polling(enc4))#(20,256,221,51)
        #dec4 = self.dec4(torch.cat([center, enc4], 1))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], mode='bilinear', align_corners=True), enc4], 1))#(20,128,221,51)
        #dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], mode='bilinear',align_corners=True), enc3], 1))#(20,64,221,51)
        #dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], mode='bilinear',align_corners=True), enc2], 1))#(20,32,221,51)
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], mode='bilinear',align_corners=True), enc1], 1))
        final = self.final(dec1)
        return final


class OperatorBlock2d(nn.Module):
    def __init__(self, in_width, out_width,modes_x, modes_y, non_lin=True):
        super(OperatorBlock2d, self).__init__()
        self.non_lin = non_lin
        self.conv = SpectralConv2d(in_width, out_width, modes_x, modes_y)
        self.mlp = MLP(in_width, out_width, out_width)
        self.w = nn.Conv2d(in_width, out_width, 1)
        self.blur=GaussianBlurLayer()

    def forward(self, x):
        x0 = F.tanh(x)
        x0 = self.blur(x0)
        x0 = self.conv(x0)
        x0 = self.mlp(x0)
        x1 = self.w(x)
        x = x0 + x1 + x
        if self.non_lin:
            x = F.gelu(x)
        return x


class DUFNO2d(nn.Module):
    def __init__(self, width, modes_x, modes_y):
        super(DUFNO2d, self).__init__()
        self.padding = 8

        self.p = nn.Linear(3, width)

        self.unet = UNet(width, width)

        self.conv0 = OperatorBlock2d(width, width, modes_x, modes_y)

        self.conv1 = OperatorBlock2d(width, width, modes_x, modes_y)

        self.conv2 = OperatorBlock2d(width, width, modes_x, modes_y)

        self.conv4 = OperatorBlock2d(width, width, modes_x, modes_y)

        self.conv5 = OperatorBlock2d(2 * width, 2 * width, modes_x, modes_y)

        self.conv6 = OperatorBlock2d(3 * width, 3 * width, modes_x, modes_y)

        self.q = MLP(3 * width, 1, 4 * width)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  # (20,221,51,4)
        x = self.p(x)  # (20,221,51,32)
        x = x.permute(0, 3, 1, 2)  # (20,32,221,51)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x = self.unet(x)

        x_c0 = self.conv0(x)

        x_c1 = self.conv1(x_c0)

        x_c2 = self.conv2(x_c1)

        x_c4 = self.conv4(x_c2)
        #x_c4 = torch.cat((x_c4, x_c1), dim=1)  # (20,64,85,85)
        x_c4 = torch.cat((F.interpolate(x_c4, x_c1.size()[-2:], mode='bilinear', align_corners=True), x_c1),dim=1)  #(20,256,85,85)

        x_c5 = self.conv5(x_c4)
        #x_c5 = torch.cat((x_c5, x_c0), dim=1)  # (20,96,85,85)
        x_c5 = torch.cat((F.interpolate(x_c5, x_c0.size()[-2:], mode='bilinear', align_corners=True), x_c0),dim=1)  #(20,256,85,85)
        x_c6 = self.conv6(x_c5)

        x_c6 = x_c6[..., :-self.padding, :-self.padding]
        x_out = self.q(x_c6)
        x_out = x_out.permute(0, 2, 3, 1)

        return x_out

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
#  configurations
################################################################
TRAIN_PATH = "/data/Darcy_rectangular_PWC/piececonst_r421_N1024_smooth1.mat"
TEST_PATH = "/data/Darcy_rectangular_PWC/piececonst_r421_N1024_smooth2.mat"

ntrain = 1000
ntest = 200

r = 3
h = int(((421 - 1) / r) + 1)
s = h

r1 = 1
h1 = int(((421 - 1) / r1) + 1)
s1 = h1

batch_size = 5

width = 32
modes_x = 12
modes_y = 12

learning_rate = 0.001
weight_decay = 1e-4
epochs = 500
step_size=100
iterations = epochs * (ntrain // batch_size)
################################################################
# load data and data normalization
################################################################
reader = MatReader(TRAIN_PATH)  # 每个数据集分辨率为(1024，421，421)
x_train = reader.read_field('coeff')[:ntrain, ::r, ::r][:, :s, :s]  # 两次切片操作：(1024，421，421)->(1000,85,85)->(1000,85,85)
y_train = reader.read_field('sol')[:ntrain, ::r, ::r][:, :s, :s]

reader.load_file(TEST_PATH)
x_test = reader.read_field('coeff')[:ntest, ::r1, ::r1][:, :s1, :s1]
y_test = reader.read_field('sol')[:ntest, ::r1, ::r1][:, :s1, :s1]

x_normalizer = UnitGaussianNormalizer(x_train)  # 对训练集归一化处理：单位高斯归一化
x_normalizer1 = UnitGaussianNormalizer(x_test)  # 对训练集归一化处理：单位高斯归一化
x_train = x_normalizer.encode(x_train)  # 将数据转化为服从正态分布
x_test = x_normalizer1.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_normalizer1 = UnitGaussianNormalizer(y_test)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain, s, s, 1)  # 将每个2d张量转化为列向量(1000,85,85,1)
x_test = x_test.reshape(ntest, s1, s1, 1)


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)
################################################################
# model definition
################################################################
model = DUFNO2d(width, modes_x, modes_y).cuda()
print(count_params(model))
################################################################
# training and evaluation
################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
y_normalizer1.cuda()
best_test_l2 = 1.0
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_mse = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()  # x(20,85,85,1)  y(20,85,85)

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s, s)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()

        optimizer.step()
        scheduler.step()

        train_mse += mse.item()
        train_l2 += loss.item()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(batch_size, s1, s1)
            out = y_normalizer1.decode(out)

            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain  # 每一代训练集上的平均损失
    test_l2 /= ntest  # 每一代测试集上的平均损失
    if test_l2 < best_test_l2:
        best_test_l2 = test_l2
    t2 = default_timer()  # 获取结束时间
    print('Epoch-{}, Time-{}, Train-Mse-{},Train-L2-{}, Test-L2-{}, Best-Test-L2-{}'
          .format(ep + 1, t2 - t1, train_mse, train_l2, test_l2, best_test_l2))
    with open('/mnt/68213B282F65B10C/LiuH/code/u-shape/uffno_filter/result/darcy/零样本超分辨率/darcy_141/darcy_141_421.txt', 'a') as file:
        print('Epoch-{}, Time-{}, Train-Mse-{},Train-L2-{}, Test-L2-{}, Best-Test-L2-{}'
              .format(ep + 1, t2 - t1, train_mse, train_l2, test_l2, best_test_l2),file=file)

    if (ep+1==1) or ((ep+1)%step_size==0):
        truth = y[0].squeeze().detach().cpu().numpy()
        pred = out[0].squeeze().detach().cpu().numpy()

        fig, ax = plt.subplots(ncols=3,figsize=(16,16))
        plt.subplots_adjust(wspace=0.5)

        ax0=ax[0].imshow(truth, cmap='viridis')
        cax0 = fig.add_axes([ax[0].get_position().x1 + 0.01, ax[0].get_position().y0, 0.01, ax[0].get_position().height])
        plt.colorbar(ax0, cax=cax0)

        ax1=ax[1].imshow(pred, cmap='viridis')
        cax1 = fig.add_axes([ax[1].get_position().x1 + 0.01, ax[1].get_position().y0, 0.01, ax[1].get_position().height])
        cbar1 = plt.colorbar(ax1,cax=cax1)

        ax2=ax[2].imshow(pred-truth, cmap='viridis')
        cax2 = fig.add_axes([ax[2].get_position().x1 + 0.01, ax[2].get_position().y0, 0.01, ax[2].get_position().height])
        cbar2 = plt.colorbar(ax2,cax=cax2)
        plt.savefig('/mnt/68213B282F65B10C/LiuH/code/u-shape/uffno_filter/result/darcy/零样本超分辨率/darcy_141/darcy_141_421_{}.png'.format(ep+1))
        plt.show()

#torch.save(model.state_dict(), '/mnt/68213B282F65B10C/LiuH/code/u-shape/uffno_filter/result/darcy/darcy.pth')