import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
from matplotlib import pyplot as plt
from utilities3 import *

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

    def forward(self, x):
        y = self.conv0(x)
        y = F.gelu(y)
        y = self.conv1(y)
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
        #self.blur = GaussianBlurLayer()

    def forward(self, x):
        x0 = F.tanh(x)
        #x0 = self.blur(x0)
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

        self.p = nn.Linear(12, width)

        self.unet = UNet(width, width)

        self.conv0 = OperatorBlock2d(width, width,modes_x, modes_y)

        self.conv1 = OperatorBlock2d(width, width,modes_x, modes_y)

        self.conv2 = OperatorBlock2d(width, width,modes_x, modes_y)

        self.conv4 = OperatorBlock2d(width, width,modes_x, modes_y)

        self.conv5 = OperatorBlock2d(2 * width, 2 * width, modes_x, modes_y)

        self.conv6 = OperatorBlock2d(3 * width, 3 * width, modes_x, modes_y)
        self.norm1 = nn.InstanceNorm2d(width)  # 归一化层，对数据进行归一化处理
        self.norm2 = nn.InstanceNorm2d(2*width)
        self.norm3 = nn.InstanceNorm2d(3 * width)

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
# configs
################################################################
TRAIN_PATH = '/mnt/68213B282F65B10C/LiuH/data/Navier_Stokes/ns_V1e-4_N10000_T30.mat'
TEST_PATH = '/mnt/68213B282F65B10C/LiuH/data/Navier_Stokes/ns_V1e-4_N10000_T30.mat'

ntrain = 1000  # 训练数据集大小
ntest = 200
# 模型参数
modesx= 12
modesy=12
width = 20

batch_size = 20  # 小批量数目
learning_rate = 0.001  # 学习率
epochs = 500  # 训练代数
iterations = epochs * (ntrain // batch_size)  # 迭代总数

# path = 'ns_fourier_2d_time_N' + str(ntrain) + '_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
# path_model = 'model/' + path
# path_train_err = 'results/' + path + 'train.txt'
# path_test_err = 'results/' + path + 'test.txt'
# path_image = 'image/' + path

sub = 1  # 采样间隔
S = 64  # 分辨率
T_in = 10#输入序列长度
T = 20  # T=40 for V1e-3; T=20 for V1e-4; T=10 for V1e-5;
step = 1#时间步长

################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)  #(1200,64,64,20)
train_a = reader.read_field('u')[:ntrain, ::sub, ::sub, :T_in]#(1200，64，64，10)
train_u = reader.read_field('u')[:ntrain, ::sub, ::sub, T_in:T + T_in]#(1200，64，64，10)

reader = MatReader(TEST_PATH)
test_a = reader.read_field('u')[-ntest:, ::sub, ::sub, :T_in]#(200,64,64,10)
test_u = reader.read_field('u')[-ntest:, ::sub, ::sub, T_in:T + T_in]#(200,64,64,10)

print(train_u.shape)
print(test_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain, S, S, T_in)#(1200,64,64,10)
test_a = test_a.reshape(ntest, S, S, T_in)#(200,64,64,10)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size,
                                          shuffle=False)

################################################################
# model definition
################################################################
model = DUFNO2d(width,modesx,modesy).cuda()
print(count_params(model))
################################################################
# training and evaluation
################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)#优化器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)#学习率优化器：学习率衰减

myloss = LpLoss(size_average=False)#lp范数做损失函数
best_test_l2 = 1.0
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:#xx(20,64,64,10)  yy(20,64,64,10)
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]#y(20,64,64,1)
            im = model(xx)#im(20,64,64,1)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im#pred(20,64,64,1)
            else:
                pred = torch.cat((pred, im), -1)#pred(20,64,64,2)

            xx = torch.cat((xx[..., step:], im), dim=-1)#...表示省略其它维度，只对最后一个维度进行切片操作,xx(20,64,64,10)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    train_l2_step = train_l2_step / ntrain / (T / step)
    train_l2_full = train_l2_full / ntrain
    test_l2_step = test_l2_step / ntest / (T / step)
    test_l2_full = test_l2_full / ntest

    if test_l2_full < best_test_l2:
        best_test_l2 = test_l2_full

    t2 = default_timer()
    #print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
    #      test_l2_full / ntest)
    print('Epoch-{}, Time-{}, Train-L2-step-{}, Train-L2-full-{},Test-L2-step-{},Test-L2-full-{}, Best-Test-L2-full-{}'
          .format(ep + 1, t2 - t1, train_l2_step , train_l2_full ,test_l2_step , test_l2_full ,best_test_l2))
    with open('/mnt/68213B282F65B10C/LiuH/code/u-shape/uffno_filter/result/ns/ns_ve-3_T50/ns.txt', 'a') as file:
        print(
            'Epoch-{}, Time-{}, Train-L2-step-{}, Train-L2-full-{},Test-L2-step-{},Test-L2-full-{}, Best-Test-L2-full-{}'
            .format(ep + 1, t2 - t1, train_l2_step, train_l2_full, test_l2_step, test_l2_full, best_test_l2), file=file)

    if (ep+1==1) or ((ep+1)%100==0):
        # torch.save(model, '../model/plas_101'+str(ep))

        truth = yy[0].squeeze().detach().cpu().numpy()
        pred = pred[0].squeeze().detach().cpu().numpy()

        fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(20, 9))
        t0,t1,t2,t3,t4 = 0,1,3,7,9
        ax0=ax[0,0].imshow(truth[:,:,t0], cmap='viridis')
        cax0 = fig.add_axes([ax[0,0].get_position().x1 + 0.01, ax[0,0].get_position().y0, 0.007, ax[0,0].get_position().height])
        plt.colorbar(ax0, cax=cax0)
        ax1=ax[1,0].imshow(pred[:,:,t0], cmap='viridis')
        cax1 = fig.add_axes([ax[1,0].get_position().x1 + 0.01, ax[1,0].get_position().y0, 0.007, ax[1,0].get_position().height])
        plt.colorbar(ax1, cax=cax1)
        ax01=ax[2,0].imshow(pred[:,:,t0]-truth[:,:,t0], cmap='viridis')
        cax01 = fig.add_axes([ax[2,0].get_position().x1 + 0.01, ax[2,0].get_position().y0, 0.007, ax[2,0].get_position().height])
        plt.colorbar(ax01, cax=cax01)

        ax2=ax[0,1].imshow(truth[:,:,t1], cmap='viridis')
        cax2 = fig.add_axes([ax[0,1].get_position().x1 + 0.01, ax[0,1].get_position().y0, 0.007, ax[0,1].get_position().height])
        plt.colorbar(ax2, cax=cax2)
        ax3=ax[1,1].imshow(pred[:,:,t1], cmap='viridis')
        cax3 = fig.add_axes([ax[1,1].get_position().x1 + 0.01, ax[1,1].get_position().y0, 0.007, ax[1,1].get_position().height])
        plt.colorbar(ax3, cax=cax3)
        ax23=ax[2,1].imshow(pred[:,:,t1]-truth[:,:,t1], cmap='viridis')
        cax23 = fig.add_axes([ax[2,1].get_position().x1 + 0.01, ax[2,1].get_position().y0, 0.007, ax[2,1].get_position().height])
        plt.colorbar(ax23, cax=cax23)

        ax4=ax[0,2].imshow(truth[:,:,t2], cmap='viridis')
        cax4 = fig.add_axes([ax[0,2].get_position().x1 + 0.01, ax[0,2].get_position().y0, 0.007, ax[0,2].get_position().height])
        plt.colorbar(ax4, cax=cax4)
        ax5=ax[1,2].imshow(pred[:,:,t2], cmap='viridis')
        cax5 = fig.add_axes([ax[1,2].get_position().x1 + 0.01, ax[1,2].get_position().y0, 0.007, ax[1,2].get_position().height])
        plt.colorbar(ax5, cax=cax5)
        ax45=ax[2,2].imshow(pred[:,:,t2]-truth[:,:,t2], cmap='viridis')
        cax45 = fig.add_axes([ax[2,2].get_position().x1 + 0.01, ax[2,2].get_position().y0, 0.007, ax[2,2].get_position().height])
        plt.colorbar(ax45, cax=cax45)

        ax6=ax[0,3].imshow(truth[:,:,t3], cmap='viridis')
        cax6 = fig.add_axes([ax[0,3].get_position().x1 + 0.01, ax[0,3].get_position().y0, 0.007, ax[0,3].get_position().height])
        plt.colorbar(ax6, cax=cax6)
        ax7=ax[1,3].imshow(pred[:,:,t3], cmap='viridis')
        cax7 = fig.add_axes([ax[1,3].get_position().x1 + 0.01, ax[1,3].get_position().y0, 0.007, ax[1,3].get_position().height])
        plt.colorbar(ax7, cax=cax7)
        ax67=ax[2,3].imshow(pred[:,:,t3]-truth[:,:,t3], cmap='viridis')
        cax67 = fig.add_axes([ax[2,3].get_position().x1 + 0.01, ax[2,3].get_position().y0, 0.007, ax[2,3].get_position().height])
        plt.colorbar(ax67, cax=cax67)

        ax8=ax[0,4].imshow(truth[:,:,t4], cmap='viridis')
        cax8 = fig.add_axes([ax[0,4].get_position().x1 + 0.01, ax[0,4].get_position().y0, 0.007, ax[0,4].get_position().height])
        plt.colorbar(ax8, cax=cax8)
        ax9=ax[1,4].imshow(pred[:,:,t4], cmap='viridis')
        cax9 = fig.add_axes([ax[1,4].get_position().x1 + 0.01, ax[1,4].get_position().y0, 0.007, ax[1,4].get_position().height])
        plt.colorbar(ax9, cax=cax9)
        ax89=ax[2,4].imshow(pred[:,:,t4]-truth[:,:,t4], cmap='viridis')
        cax89 = fig.add_axes([ax[2,4].get_position().x1 + 0.01, ax[2,4].get_position().y0, 0.007, ax[2,4].get_position().height])
        plt.colorbar(ax89, cax=cax89)
        plt.savefig('/mnt/68213B282F65B10C/LiuH/code/u-shape/uffno_filter/result/ns/ns_ve-3_T50/ns_{}.png'.format(ep + 1))
        plt.show()