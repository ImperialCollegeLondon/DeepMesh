import torch
from torch import nn
import torch.nn.functional as F

def relu():
    return nn.ReLU(inplace=True)


def conv(in_channels, out_channels, kernel_size=(3,3,3), stride=(1,1,1), padding = 1, nonlinearity = relu):
    conv_layer = nn.Conv3d(in_channels = in_channels, out_channels= out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)

    nll_layer = nonlinearity()
    bn_layer = nn.BatchNorm3d(out_channels)

    layers = [conv_layer, bn_layer, nll_layer]
    return nn.Sequential(*layers)

def deconv(in_channels, out_channels, kernel_size=(3,3,3), stride=(1,1,1), padding = 1, nonlinearity = relu):
    conv_layer = nn.ConvTranspose3d(in_channels = in_channels, out_channels= out_channels, kernel_size = kernel_size, stride = stride, padding = padding, output_padding = 1, bias = False)

    nll_layer = nonlinearity()
    bn_layer = nn.BatchNorm3d(out_channels)

    layers = [conv_layer, bn_layer, nll_layer]
    return nn.Sequential(*layers)


def conv_blocks_2(in_channels, out_channels, strides=(1,1,1)):
    conv1 = conv(in_channels, out_channels, stride = strides)
    conv2 = conv(out_channels, out_channels, stride=(1,1,1))
    layers = [conv1, conv2]
    return nn.Sequential(*layers)


def conv_blocks_3(in_channels, out_channels, strides=(1,1,1)):
    conv1 = conv(in_channels, out_channels, stride = strides)
    conv2 = conv(out_channels, out_channels, stride=(1,1,1))
    conv3 = conv(out_channels, out_channels, stride=(1,1,1))
    layers = [conv1, conv2, conv3]
    return nn.Sequential(*layers)

def fullyconnect(in_features, out_features, out_channels, nonlinearity = relu):
    fc_layer = nn.Linear(in_features = in_features, out_features= out_features, bias = False)

    nll_layer = nonlinearity()
    bn_layer = nn.BatchNorm1d(out_channels)

    layers = [fc_layer, bn_layer, nll_layer]
    return nn.Sequential(*layers)

def conv_2D(in_channels, out_channels, kernel_size=3, stride=1, padding = 1, nonlinearity = relu):
    conv_layer = nn.Conv2d(in_channels = in_channels, out_channels= out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)

    nll_layer = nonlinearity()
    bn_layer = nn.BatchNorm2d(out_channels)

    layers = [conv_layer, bn_layer, nll_layer]
    return nn.Sequential(*layers)

def conv_1D(in_channels, out_channels, kernel_size=3, stride=1, padding = 1, nonlinearity = relu):
    conv_layer = nn.Conv1d(in_channels = in_channels, out_channels= out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)

    nll_layer = nonlinearity()
    bn_layer = nn.BatchNorm2d(out_channels)

    layers = [conv_layer, bn_layer, nll_layer]
    return nn.Sequential(*layers)

def deconv_2D(in_channels, out_channels, kernel_size=3, stride=1, padding = 1, nonlinearity = relu):
    conv_layer = nn.ConvTranspose2d(in_channels = in_channels, out_channels= out_channels, kernel_size = kernel_size, stride = stride, padding = padding, output_padding = 1, bias = False)

    nll_layer = nonlinearity()
    bn_layer = nn.BatchNorm2d(out_channels)

    layers = [conv_layer, bn_layer, nll_layer]
    return nn.Sequential(*layers)


def conv_blocks_2_2D(in_channels, out_channels, strides=1):
    conv1 = conv_2D(in_channels, out_channels, stride = strides)
    conv2 = conv_2D(out_channels, out_channels, stride=1)
    layers = [conv1, conv2]
    return nn.Sequential(*layers)


def conv_blocks_3_2D(in_channels, out_channels, strides=1):
    conv1 = conv_2D(in_channels, out_channels, stride = strides)
    conv2 = conv_2D(out_channels, out_channels, stride=1)
    conv3 = conv_2D(out_channels, out_channels, stride=1)
    layers = [conv1, conv2, conv3]
    return nn.Sequential(*layers)

# Flatten layer
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def generate_grid(x, offset):
    x_shape = x.size()
    grid_d, grid_w, grid_h = torch.meshgrid([torch.linspace(-1, 1, x_shape[2]), torch.linspace(-1, 1, x_shape[3]), torch.linspace(-1, 1, x_shape[4])])  # (h, w, h)
    grid_d = grid_d.cuda().float()
    grid_w = grid_w.cuda().float()
    grid_h = grid_h.cuda().float()

    grid_d = nn.Parameter(grid_d, requires_grad=False)
    grid_w = nn.Parameter(grid_w, requires_grad=False)
    grid_h = nn.Parameter(grid_h, requires_grad=False)

    offset_h, offset_w, offset_d = torch.split(offset, 1, 1)
    offset_d = offset_d.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), int(x_shape[4]))  # (b*c, d, w, h)
    offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), int(x_shape[4]))  # (b*c, d, w, h)
    offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), int(x_shape[4]))  # (b*c, d, w, h)

    offset_d = grid_d + offset_d
    offset_w = grid_w + offset_w
    offset_h = grid_h + offset_h

    offsets = torch.stack((offset_h, offset_w, offset_d), 4) # should have the same order as offset
    return offsets

def transform(seg_source, loc, mode='bilinear'):
    grid = generate_grid(seg_source, loc)
    # seg_source: NCDHW
    # grid: NDHW3
    out = F.grid_sample(seg_source, grid, mode=mode, align_corners=True)
    return out



class Mesh_2d(nn.Module):
    """Deformable registration network with input from image space """
    def __init__(self, n_ch=1):
        super(Mesh_2d, self).__init__()

        self.conv1 = conv_2D(n_ch, 32)
        self.conv2 = conv_2D(32, 64)

    def forward(self, x_2ch, x_2ched, x_4ch, x_4ched):
        # x: source image; x_pred: target image;
        net = {}

        net['conv1_2ch'] = self.conv1(x_2ch)
        net['conv1_4ch'] = self.conv1(x_4ch)
        net['conv1s_2ch'] = self.conv1(x_2ched)
        net['conv1s_4ch'] = self.conv1(x_4ched)

        net['conv2_2ch'] = self.conv2(net['conv1_2ch'])
        net['conv2_4ch'] = self.conv2(net['conv1_4ch'])
        net['conv2s_2ch'] = self.conv2(net['conv1s_2ch'])
        net['conv2s_4ch'] = self.conv2(net['conv1s_4ch'])


        return net

class deformnet(nn.Module):
    """Deformable registration network with input from image space """
    def __init__(self, n_ch=64, mesh_dim=22043):
        super(deformnet, self).__init__()

        self.conv_blocks_2D = [conv_blocks_2_2D(n_ch, 64), conv_blocks_2_2D(64, 128, 2), conv_blocks_3_2D(128, 256, 2),
                            conv_blocks_3_2D(256, 512, 2), conv_blocks_3_2D(512, 512, 2)]

        self.conv_blocks_2D = nn.Sequential(*self.conv_blocks_2D)

        self.conv2d9 = conv_2D(512 * 3, 512 * 2, kernel_size=3, stride=1)
        self.conv2d10 = deconv_2D(512 * 2, 512, kernel_size=3, stride=2)
        self.conv2d10_1 = conv_2D(512, 512, kernel_size=3, stride=1)
        self.conv2d11 = deconv_2D(512, 256, kernel_size=3, stride=2)
        self.conv2d11_1 = conv_2D(256, 256, kernel_size=3, stride=1)
        self.conv2d12 = deconv_2D(256, 128, kernel_size=3, stride=2)
        self.conv2d12_1 = conv_2D(128, 128, kernel_size=3, stride=1)
        self.conv2d13 = deconv_2D(128, 64, kernel_size=3, stride=2)
        self.conv2d13_1 = conv_2D(64, 64, kernel_size=3, stride=1)

        self.conv3d17 = conv(1, 3, 3, stride=(1, 1, 1))
        self.conv3d18 = nn.Conv3d(3, 3, 1, stride=(1, 1, 1))


    def forward(self, x_saed, x_2ched, x_4ched):
        # x: source image; x_pred: target image;
        net = {}

        net['conv0_sa_ed'] = x_saed
        net['conv0_2ch_ed'] = x_2ched
        net['conv0_4ch_ed'] = x_4ched
        # 5 refers to 5 output or 5 blocks
        for i in range(5):
            net['conv%d_sa_ed' % (i + 1)] = self.conv_blocks_2D[i](net['conv%d_sa_ed' % i])
            net['conv%d_2ch_ed' % (i + 1)] = self.conv_blocks_2D[i](net['conv%d_2ch_ed' % i])
            net['conv%d_4ch_ed' % (i + 1)] = self.conv_blocks_2D[i](net['conv%d_4ch_ed' % i])


        net['concat'] = torch.cat((net['conv5_sa_ed'], net['conv5_2ch_ed'], net['conv5_4ch_ed']), 1)

        net['conv2d_0_ed'] = self.conv2d9(net['concat'])
        net['conv2d_1_ed'] = self.conv2d10_1(self.conv2d10(net['conv2d_0_ed']))
        net['conv2d_2_ed'] = self.conv2d11_1(self.conv2d11(net['conv2d_1_ed']))
        net['conv2d_3_ed'] = self.conv2d12_1(self.conv2d12(net['conv2d_2_ed']))
        net['conv2d_4_ed'] = self.conv2d13_1(self.conv2d13(net['conv2d_3_ed']))
        net['conv3d_def_ed'] = net['conv2d_4_ed'].unsqueeze(1)
        net['out_def_ed'] = torch.tanh(self.conv3d18(self.conv3d17(net['conv3d_def_ed'])))




        return net