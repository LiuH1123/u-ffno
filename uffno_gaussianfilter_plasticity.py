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
    xx, yy, zz = np.meshgrid(ax, ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy) + np.square(zz)) / np.square(sigma))
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
        kernel = self.weight.expand(x.size(1), 1, self.weight.size(2), self.weight.size(3),self.weight.size(4))
        return F.conv3d(x, kernel, padding=self.weight.size(2) // 2, groups=x.size(1))


class SpectralConv3d(nn.Module):
    def __init__(self, in_dim, out_dim, modes_x, modes_y, modes_z, fourier_weight=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z

        self.fourier_weight = fourier_weight
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_x, modes_y, modes_z]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)


    def forward(self, x):
        #x = rearrange(x, 'b s1 s2 s3 i -> b i s1 s2 s3')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        B, I, S1, S2, S3 = x.shape

        # # # Dimesion Z # # #
        x_ftz = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_ftz.new_zeros(B, I, S1, S2, S3 // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :, :, :self.modes_z] = torch.einsum(
            "bixyz,ioz->boxyz",
            x_ftz[:, :, :, :, :self.modes_z],
            torch.view_as_complex(self.fourier_weight[2]))

        xz = torch.fft.irfft(out_ft, n=S3, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_fty.new_zeros(B, I, S1, S2 // 2 + 1, S3)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :, :self.modes_y, :] = torch.einsum(
            "bixyz,ioy->boxyz",
            x_fty[:, :, :, :self.modes_y, :],
            torch.view_as_complex(self.fourier_weight[1]))

        xy = torch.fft.irfft(out_ft, n=S2, dim=-2, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-3, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, S1 // 2 + 1, S2, S3)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :self.modes_x, :, :] = torch.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, :self.modes_x, :, :],
            torch.view_as_complex(self.fourier_weight[0]))

        xx = torch.fft.irfft(out_ft, n=S1, dim=-3, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy + xz

        #x = rearrange(x, 'b i s1 s2 s3 -> b s1 s2 s3 i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x


class Residual(nn.Module):
    def __init__(self, in_width,out_width):
        super(Residual, self).__init__()
        self.conv0 = nn.Conv3d(in_width, out_width, 1)
        self.conv1 = nn.Conv3d(out_width, out_width, 1)
        self.bn0 = nn.BatchNorm3d(out_width)
        self.bn1 = nn.BatchNorm3d(out_width)

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
        self.mlp1 = nn.Conv3d(in_width, mid_width, 1)
        self.mlp2 = nn.Conv3d(mid_width, out_width, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, polling=True, bn=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU(),
        ]
        if dropout:
            layers.append(nn.Dropout(0.25))
        self.encode = nn.Sequential(*layers)
        self.pool = None
        if polling:
            self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bn=False, dropout=False):
        super(_DecoderBlock, self).__init__()
        if dropout:
            self.decode = nn.Sequential(
                nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
                nn.GELU(),
                nn.Conv3d(middle_channels, middle_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.ConvTranspose3d(middle_channels, out_channels, kernel_size=2, stride=2),
            )
        else:
            self.decode = nn.Sequential(
                nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
                nn.GELU(),
                nn.Conv3d(middle_channels, middle_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
                nn.GELU(),
                nn.ConvTranspose3d(middle_channels, out_channels, kernel_size=2, stride=2),
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
        self.polling = nn.MaxPool3d(kernel_size=2, stride=2)
        self.center = _DecoderBlock(256, 512, 256, bn=bn)
        self.dec4 = _DecoderBlock(512, 256, 128, bn=bn)
        self.dec3 = _DecoderBlock(256, 128, 64, bn=bn)
        self.dec2 = _DecoderBlock(128, 64, 32, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(32) if bn else nn.GroupNorm(32, 64),
            nn.GELU()
        )
        self.final = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)#(20,32,221,51)
        enc2 = self.enc2(enc1)#(20,64,221,51)
        enc3 = self.enc3(enc2)#(20,128,221,51)
        enc4 = self.enc4(enc3)#(20,256,221,51)
        center = self.center(self.polling(enc4))#(20,256,221,51)
        #dec4 = self.dec4(torch.cat([center, enc4], 1))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-3:], mode='trilinear', align_corners=True), enc4], 1))#(20,128,221,51)
        #dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-3:], mode='trilinear',align_corners=True), enc3], 1))#(20,64,221,51)
        #dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-3:], mode='trilinear',align_corners=True), enc2], 1))#(20,32,221,51)
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-3:], mode='trilinear',align_corners=True), enc1], 1))
        final = self.final(dec1)
        return final


class OperatorBlock3d(nn.Module):
    def __init__(self, in_width, out_width,modes_x, modes_y, modes_z,non_lin=True):
        super(OperatorBlock3d, self).__init__()
        self.non_lin = non_lin
        self.conv = SpectralConv3d(in_width, out_width, modes_x, modes_y, modes_z)
        self.mlp = MLP(in_width, out_width, out_width)
        self.w = nn.Conv3d(in_width, out_width, 1)
        self.res=Residual(out_width,out_width)
        #self.blur=GaussianBlurLayer()

    def forward(self, x):
        #x0 = F.tanh(x)
        #x0 = self.blur(x)
        x0 = self.conv(x)
        x0 = self.mlp(x0)
        x1 = self.w(x)
        x = x0 + x1 + x
        x=self.res(x)
        if self.non_lin:
            x = F.gelu(x)
        return x


class UFFNO3d(nn.Module):
    def __init__(self, width, modes_x, modes_y, modes_z):
        super(UFFNO3d, self).__init__()
        self.padding = 5

        self.p = nn.Linear(4, width)

        self.unet = UNet(width, width)

        #self.blur = GaussianBlurLayer()

        self.conv0 = OperatorBlock3d(width, width,modes_x, modes_y,modes_z)

        self.conv1 = OperatorBlock3d(width, width,modes_x, modes_y,modes_z)

        self.conv2 = OperatorBlock3d(width, width,modes_x, modes_y,modes_z)

        self.conv4 = OperatorBlock3d(width, width,modes_x, modes_y,modes_z)

        self.conv5 = OperatorBlock3d(2 * width, 2 * width, modes_x, modes_y,modes_z)

        self.conv6 = OperatorBlock3d(3 * width, 3 * width, modes_x, modes_y,modes_z)

        self.w1 = nn.Conv3d(width,width,1)

        self.w2 = nn.Conv3d(width, width, 1)

        self.q = MLP(3 * width, 4,  4*width)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  # (20,221,51,4)
        x = self.p(x)  # (20,221,51,32)
        x = x.permute(0, 4, 1, 2, 3)  # (20,32,221,51)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x = self.unet(x)

        x_c0 = self.conv0(x)

        x_c1 = self.conv1(x_c0)

        x_c2 = self.conv2(x_c1)

        x_c4 = self.conv4(x_c2)
        x_c4 = torch.cat((F.interpolate(x_c4, x_c1.size()[-3:], mode='trilinear', align_corners=True), self.w1(x_c1)),dim=1)  #(20,256,85,85)

        x_c5 = self.conv5(x_c4)
        x_c5 = torch.cat((F.interpolate(x_c5, x_c0.size()[-3:], mode='trilinear', align_corners=True), self.w2(x_c0)),dim=1)  #(20,256,85,85)

        x_c6 = self.conv6(x_c5)

        x_c6 = x_c6[..., :-self.padding, :-self.padding]
        x_out = self.q(x_c6)
        x_out = x_out.permute(0, 2, 3, 4, 1)

        return x_out

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
################################################################
# configs
################################################################
DATA_PATH = '/mnt/68213B282F65B10C/LiuH/data/plasticity/plas_N987_T20.mat'

N = 987
ntrain = 900
ntest = 80

batch_size = 10
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5
iterations = epochs * (ntrain // batch_size)

modes_x = 12
modes_y = 12
modes_z = 12
width = 32
out_dim = 4

s1 = 101
s2 = 31
t = 20

r1 = 1
r2 = 1
s1 = int(((s1 - 1) / r1) + 1)
s2 = int(((s2 - 1) / r2) + 1)

################################################################
# load data and data normalization
################################################################
reader = MatReader(DATA_PATH)
x_train = reader.read_field('input')[:ntrain, ::r1][:, :s1].reshape(ntrain, s1, 1, 1, 1).repeat(1, 1, s2, t, 1)
y_train = reader.read_field('output')[:ntrain, ::r1, ::r2][:, :s1, :s2]
reader.load_file(DATA_PATH)
x_test = reader.read_field('input')[-ntest:, ::r1][:, :s1].reshape(ntest, s1, 1, 1, 1).repeat(1, 1, s2, t, 1)#(900,101,31,20,1)
y_test = reader.read_field('output')[-ntest:, ::r1, ::r2][:, :s1, :s2]#(80,101,31)
print(x_train.shape, y_train.shape)
x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)
y_normalizer.cuda()

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)

################################################################
# training and evaluation
################################################################
model=UFFNO3d(width,modes_x,modes_y,modes_z).cuda()
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False, p=2)

best_test_l2 = 1.0
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    #train_reg = 0
    train_mse = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s1, s2, t, out_dim)

        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        mse = F.mse_loss(out.contiguous().view(batch_size, -1), y.contiguous().view(batch_size, -1), reduction='mean')
        loss = myloss(out.contiguous().view(batch_size, -1), y.contiguous().view(batch_size, -1))
        loss.backward()

        optimizer.step()
        scheduler.step()
        train_mse += mse.item()
        train_l2 += loss.item()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:#x(4,101,31,20,1),y(4,101,31,20,4)
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(batch_size, s1, s2, t, out_dim)#(4,101,31,20,4)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.contiguous().reshape(batch_size, -1), y.contiguous().view(batch_size, -1)).item()

    train_l2 /= ntrain
    #train_reg /= ntrain
    test_l2 /= ntest
    if test_l2 < best_test_l2:
        best_test_l2 = test_l2
    t2 = default_timer()
    print('Epoch-{}, Time-{}, Train-Mse-{}, Train-L2-{}, Test-L2-{}, Best-Test-L2-{}'
          .format(ep + 1, t2 - t1, train_mse, train_l2, test_l2, best_test_l2))
    with open('/mnt/68213B282F65B10C/LiuH/code/u-shape/uffno_filter/result/plasticity/plasticity.txt', 'a') as file:
        print('Epoch-{}, Time-{}, Train-Mse-{}, Train-L2-{}, Test-L2-{}, Best-Test-L2-{}'
              .format(ep + 1, t2 - t1, train_mse, train_l2, test_l2, best_test_l2),file=file)

    if (ep + 1 == 1) or ((ep + 1) % 50 == 0):
        truth = y[0].squeeze().detach().cpu().numpy()
        pred = out[0].squeeze().detach().cpu().numpy()
        ZERO = torch.zeros(s1, s2)
        truth_du = np.linalg.norm(truth[:, :, :, 2:], axis=-1)
        pred_du = np.linalg.norm(pred[:, :, :, 2:], axis=-1)

        lims = dict(cmap='RdBu_r', vmin=truth_du.min(), vmax=truth_du.max())
        fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(20, 9))
        plt.subplots_adjust(wspace=0.5)
        t0, t1, t2, t3, t4 = 0, 4, 9, 14, 19
        ax0 = ax[0, 0].scatter(truth[:, :, 0, 0], truth[:, :, 0, 1], 10, truth_du[:, :, 0], **lims)
        cax0 = fig.add_axes(
            [ax[0, 0].get_position().x1 + 0.01, ax[0, 0].get_position().y0, 0.007, ax[0, 0].get_position().height])
        plt.colorbar(ax0, cax=cax0)
        ax1 = ax[1, 0].scatter(pred[:, :, 0, 0], pred[:, :, 0, 1], 10, pred_du[:, :, 0], **lims)
        cax1 = fig.add_axes(
            [ax[1, 0].get_position().x1 + 0.01, ax[1, 0].get_position().y0, 0.007, ax[1, 0].get_position().height])
        plt.colorbar(ax1, cax=cax1)
        ax01 = ax[2, 0].scatter(truth[:, :, 0, 0], truth[:, :, 0, 1], 10, pred_du[:, :, 0] - truth_du[:, :, 0], **lims)
        cax01 = fig.add_axes(
            [ax[2, 0].get_position().x1 + 0.01, ax[2, 0].get_position().y0, 0.007, ax[2, 0].get_position().height])
        plt.colorbar(ax01, cax=cax01)

        ax2 = ax[0, 1].scatter(truth[:, :, 4, 0], truth[:, :, 4, 1], 10, truth_du[:, :, 4], **lims)
        cax2 = fig.add_axes(
            [ax[0, 1].get_position().x1 + 0.01, ax[0, 1].get_position().y0, 0.007, ax[0, 1].get_position().height])
        plt.colorbar(ax2, cax=cax2)
        ax3 = ax[1, 1].scatter(pred[:, :, 4, 0], pred[:, :, 4, 1], 10, pred_du[:, :, 4], **lims)
        cax3 = fig.add_axes(
            [ax[1, 1].get_position().x1 + 0.01, ax[1, 1].get_position().y0, 0.007, ax[1, 1].get_position().height])
        plt.colorbar(ax3, cax=cax3)
        ax23 = ax[2, 1].scatter(truth[:, :, 4, 0], truth[:, :, 4, 1], 10, pred_du[:, :, 4] - truth_du[:, :, 4], **lims)
        cax23 = fig.add_axes(
            [ax[2, 1].get_position().x1 + 0.01, ax[2, 1].get_position().y0, 0.007, ax[2, 1].get_position().height])
        plt.colorbar(ax23, cax=cax23)

        ax4 = ax[0, 2].scatter(truth[:, :, 9, 0], truth[:, :, 9, 1], 10, truth_du[:, :, 9], **lims)
        cax4 = fig.add_axes(
            [ax[0, 2].get_position().x1 + 0.01, ax[0, 2].get_position().y0, 0.007, ax[0, 2].get_position().height])
        plt.colorbar(ax4, cax=cax4)
        ax5 = ax[1, 2].scatter(pred[:, :, 9, 0], pred[:, :, 9, 1], 10, pred_du[:, :, 9], **lims)
        cax5 = fig.add_axes(
            [ax[1, 2].get_position().x1 + 0.01, ax[1, 2].get_position().y0, 0.007, ax[1, 2].get_position().height])
        plt.colorbar(ax5, cax=cax5)
        ax45 = ax[2, 2].scatter(truth[:, :, 9, 0], truth[:, :, 9, 1], 10, pred_du[:, :, 9] - truth_du[:, :, 9], **lims)
        cax01 = fig.add_axes(
            [ax[2, 2].get_position().x1 + 0.01, ax[2, 2].get_position().y0, 0.007, ax[2, 2].get_position().height])
        plt.colorbar(ax01, cax=cax01)

        ax6 = ax[0, 3].scatter(truth[:, :, 14, 0], truth[:, :, 14, 1], 10, truth_du[:, :, 14], **lims)
        cax6 = fig.add_axes(
            [ax[0, 3].get_position().x1 + 0.01, ax[0, 3].get_position().y0, 0.007, ax[0, 3].get_position().height])
        plt.colorbar(ax6, cax=cax6)
        ax7 = ax[1, 3].scatter(pred[:, :, 14, 0], pred[:, :, 14, 1], 10, pred_du[:, :, 14], **lims)
        cax7 = fig.add_axes(
            [ax[1, 3].get_position().x1 + 0.01, ax[1, 3].get_position().y0, 0.007, ax[1, 3].get_position().height])
        plt.colorbar(ax0, cax=cax7)
        ax67 = ax[2, 3].scatter(truth[:, :, 14, 0], truth[:, :, 14, 1], 10, pred_du[:, :, 14] - truth_du[:, :, 14],
                                **lims)
        cax01 = fig.add_axes(
            [ax[2, 3].get_position().x1 + 0.01, ax[2, 3].get_position().y0, 0.007, ax[2, 3].get_position().height])
        plt.colorbar(ax01, cax=cax01)

        ax8 = ax[0, 4].scatter(truth[:, :, 19, 0], truth[:, :, 19, 1], 10, truth_du[:, :, 19], **lims)
        cax8 = fig.add_axes(
            [ax[0, 4].get_position().x1 + 0.01, ax[0, 4].get_position().y0, 0.007, ax[0, 4].get_position().height])
        plt.colorbar(ax8, cax=cax8)
        ax9 = ax[1, 4].scatter(pred[:, :, 19, 0], pred[:, :, 19, 1], 10, pred_du[:, :, 19], **lims)
        cax9 = fig.add_axes(
            [ax[1, 4].get_position().x1 + 0.01, ax[1, 4].get_position().y0, 0.007, ax[1, 4].get_position().height])
        plt.colorbar(ax9, cax=cax9)
        ax89 = ax[2, 4].scatter(truth[:, :, 19, 0], truth[:, :, 19, 1], 10, pred_du[:, :, 19] - truth_du[:, :, 19],
                                **lims)
        cax89 = fig.add_axes(
            [ax[2, 4].get_position().x1 + 0.01, ax[2, 4].get_position().y0, 0.007, ax[2, 4].get_position().height])
        plt.colorbar(ax89, cax=cax89)
        plt.savefig('/mnt/68213B282F65B10C/LiuH/code/u-shape/uffno_filter/result/plasticity/plasticity_{}.png'.format(ep + 1))
        plt.show()