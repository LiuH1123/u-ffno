import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *
from matplotlib import pyplot as plt

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


class UFFNO2d(nn.Module):
    def __init__(self, width, modes_x, modes_y):
        super(UFFNO2d, self).__init__()
        self.padding = 8

        self.p = nn.Linear(4, width)

        self.unet = UNet(width, width)

        self.conv0 = OperatorBlock2d(width, width,modes_x, modes_y)

        self.conv1 = OperatorBlock2d(width, width,modes_x, modes_y)

        self.conv2 = OperatorBlock2d(width, width,modes_x, modes_y)

        self.conv4 = OperatorBlock2d(width, width,modes_x, modes_y)

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
        x_c4 = torch.cat((F.interpolate(x_c4, x_c1.size()[-2:], mode='bilinear', align_corners=True), x_c1),dim=1)  #(20,256,85,85)

        x_c5 = self.conv5(x_c4)
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
INPUT_X = "/mnt/68213B282F65B10C/LiuH/data/pipe/Pipe_X.npy"
INPUT_Y = "/mnt/68213B282F65B10C/LiuH/data/pipe/Pipe_Y.npy"
OUTPUT_Sigma = "/mnt/68213B282F65B10C/LiuH/data/pipe/Pipe_Q.npy"

ntrain = 1000
ntest = 200
N = 1200

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5
iterations = epochs * (ntrain // batch_size)

modes_x = 12
modes_y=12
width = 32

r1 = 1
r2 = 1
s1 = int(((129 - 1) / r1) + 1)
s2 = int(((129 - 1) / r2) + 1)

################################################################
# load data and data normalization
################################################################
inputX = np.load(INPUT_X)
inputX = torch.tensor(inputX, dtype=torch.float)  # (2310,129,129)
inputY = np.load(INPUT_Y)
inputY = torch.tensor(inputY, dtype=torch.float)  # (2310,129,129)
input = torch.stack([inputX, inputY], dim=-1)  # (2310,129,129,2)

output = np.load(OUTPUT_Sigma)[:, 0]
output = torch.tensor(output, dtype=torch.float)  # (2310,129,129)

x_train = input[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]  # (1000,129,129,2)
y_train = output[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]  # (1000,129,129)
x_test = input[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
y_test = output[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
x_train = x_train.reshape(ntrain, s1, s2, 2)
x_test = x_test.reshape(ntest, s1, s2, 2)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)
test_loader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1,
                                           shuffle=False)

################################################################
# training and evaluation
################################################################
model = UFFNO2d(width, modes_x, modes_y).cuda()
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)
best_test_l2 = 1.0
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_mse = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)

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

            out = model(x)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest
    if test_l2 < best_test_l2:
        best_test_l2 = test_l2
    t2 = default_timer()
    print('Epoch-{}, Time-{}, Train-Mse-{}, Train-L2-{}, Test-L2-{}, Best-Test-L2-{}'
          .format(ep + 1, t2 - t1, train_mse, train_l2, test_l2, best_test_l2))
    with open('/mnt/68213B282F65B10C/LiuH/code/u-shape/uffno_filter/result/pipe/pipe.txt', 'a') as file:
        print('Epoch-{}, Time-{}, Train-Mse-{},Train-L2-{}, Test-L2-{}, Best-Test-L2-{}'
              .format(ep + 1, t2 - t1, train_mse, train_l2, test_l2, best_test_l2),file=file)

    if (ep+1==1) or ((ep+1)%step_size==0):
         #torch.save(model, '../model/pipe_' + str(ep))
        X = x[0, :, :, 0].squeeze().detach().cpu().numpy()
        Y = x[0, :, :, 1].squeeze().detach().cpu().numpy()
        truth = y[0].squeeze().detach().cpu().numpy()
        pred = out[0].squeeze().detach().cpu().numpy()

        fig, ax = plt.subplots(nrows=3, figsize=(16, 16))
        ax0=ax[0].pcolormesh(X, Y, truth, shading='gouraud')
        cax0 = fig.add_axes([ax[0].get_position().x1 + 0.01, ax[0].get_position().y0, 0.01, ax[0].get_position().height])
        plt.colorbar(ax0, cax=cax0)
        ax1=ax[1].pcolormesh(X, Y, pred, shading='gouraud')
        cax1 = fig.add_axes([ax[1].get_position().x1 + 0.01, ax[1].get_position().y0, 0.01, ax[1].get_position().height])
        plt.colorbar(ax1, cax=cax1)
        ax2=ax[2].pcolormesh(X, Y, pred-truth, shading='gouraud')
        cax2 = fig.add_axes([ax[2].get_position().x1 + 0.01, ax[2].get_position().y0, 0.01, ax[2].get_position().height])
        plt.colorbar(ax2, cax=cax2)
        plt.savefig('/mnt/68213B282F65B10C/LiuH/code/u-shape/uffno_filter/result/pipe/pipe_{}.png'.format(ep+1))
        plt.show()
