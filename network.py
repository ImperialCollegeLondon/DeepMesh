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
    # when input is 5D the mode='bilinear' is used as trilinear
    out = F.grid_sample(seg_source, grid, mode=mode, align_corners=True)
    return out


class MotionMesh_25d(nn.Module):
    """Deformable registration network with input from image space """
    def __init__(self, n_ch=64):
        super(MotionMesh_25d, self).__init__()

        self.conv_blocks_2D = [conv_blocks_2_2D(n_ch, 64), conv_blocks_2_2D(64, 128, 2), conv_blocks_3_2D(128, 256, 2),
                            conv_blocks_3_2D(256, 512, 2), conv_blocks_3_2D(512, 512, 2)]
        self.conv_list_2D = []
        for in_filters in [128, 256, 512, 1024, 1024]:
            self.conv_list_2D += [conv_2D(in_filters, 64)]

        self.conv_blocks_2D = nn.Sequential(*self.conv_blocks_2D)
        self.conv_list_2D = nn.Sequential(*self.conv_list_2D)

        self.conv6 = conv_2D(64 * 15, 64, 1, 1, 0)
        self.conv7 = conv_2D(64, 64, 1, 1, 0)
        self.conv3d9 = conv(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv3d10 = conv(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2))
        self.conv3d10_1 = conv(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv3d11 = conv(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2))
        self.conv3d11_1 = conv(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv3d12 = conv(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2))
        self.conv3d12_1 = conv(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv3d13 = deconv(256, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2))
        self.conv3d13_1 = conv(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv3d14 = deconv(128, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2))
        self.conv3d14_1 = conv(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv3d15 = deconv(64, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2))
        self.conv3d15_1 = conv(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv3d16 = nn.Conv3d(32, 3, 1, stride=(1, 1, 1))

        self.conv2d9 = conv_2D(512*3, 512*2, kernel_size=3, stride=1)
        self.conv2d10 = deconv_2D(512*2, 512, kernel_size=3, stride=2)
        self.conv2d10_1 = conv_2D(512, 512, kernel_size=3, stride=1)
        self.conv2d11 = deconv_2D(512, 256, kernel_size=3, stride=2)
        self.conv2d11_1 = conv_2D(256, 256, kernel_size=3, stride=1)
        self.conv2d12 = deconv_2D(256, 128, kernel_size=3, stride=2)
        self.conv2d12_1 = conv_2D(128, 128, kernel_size=3, stride=1)
        self.conv2d13 = deconv_2D(128, 64, kernel_size=3, stride=2)
        self.conv2d13_1 = conv_2D(64, 64, kernel_size=3, stride=1)

        self.conv3d17 = nn.Conv3d(1, 2, 1, stride=(1, 1, 1))
        self.conv3d18 = nn.Conv3d(2, 2, 1,stride=(1, 1, 1))


    def forward(self, x_sa, x_saed, x_2ch, x_2ched, x_4ch, x_4ched):
        # x: source image; x_pred: target image;
        net = {}
        net['conv0_sa'] = x_sa
        net['conv0_sa_ed'] = x_saed
        net['conv0_2ch'] = x_2ch
        net['conv0_2ch_ed'] = x_2ched
        net['conv0_4ch'] = x_4ch
        net['conv0_4ch_ed'] = x_4ched
        # 5 refers to 5 output or 5 blocks
        for i in range(5):
            net['conv%d_sa' % (i + 1)] = self.conv_blocks_2D[i](net['conv%d_sa' % i])
            net['conv%d_sa_ed' % (i + 1)] = self.conv_blocks_2D[i](net['conv%d_sa_ed' % i])
            net['conv%d_2ch' % (i + 1)] = self.conv_blocks_2D[i](net['conv%d_2ch' % i])
            net['conv%d_2ch_ed' % (i + 1)] = self.conv_blocks_2D[i](net['conv%d_2ch_ed' % i])
            net['conv%d_4ch' % (i + 1)] = self.conv_blocks_2D[i](net['conv%d_4ch' % i])
            net['conv%d_4ch_ed' % (i + 1)] = self.conv_blocks_2D[i](net['conv%d_4ch_ed' % i])

            net['concat%d_sa' % (i + 1)] = torch.cat((net['conv%d_sa' % (i + 1)], net['conv%d_sa_ed' % (i + 1)]), 1)
            net['concat%d_2ch' % (i + 1)] = torch.cat((net['conv%d_2ch' % (i + 1)], net['conv%d_2ch_ed' % (i + 1)]), 1)
            net['concat%d_4ch' % (i + 1)] = torch.cat((net['conv%d_4ch' % (i + 1)], net['conv%d_4ch_ed' % (i + 1)]), 1)

            net['out%d_sa' % (i + 1)] = self.conv_list_2D[i](net['concat%d_sa' % (i + 1)])
            net['out%d_2ch' % (i + 1)] = self.conv_list_2D[i](net['concat%d_2ch' % (i + 1)])
            net['out%d_4ch' % (i + 1)] = self.conv_list_2D[i](net['concat%d_4ch' % (i + 1)])
            if i > 0:
                # upsample DHW dimension
                net['out%d_sa_up' % (i + 1)] = F.interpolate(net['out%d_sa' % (i + 1)], scale_factor=2 ** i, mode='bilinear', align_corners=True)
                net['out%d_2ch_up' % (i + 1)] = F.interpolate(net['out%d_2ch' % (i + 1)], scale_factor=2 ** i, mode='bilinear', align_corners=True)
                net['out%d_4ch_up' % (i + 1)] = F.interpolate(net['out%d_4ch' % (i + 1)], scale_factor=2 ** i, mode='bilinear', align_corners=True)



        # output: net['out1_sa'], net['out2_sa_up'], net['out3_sa_up'], net['out4_sa_up'], net['out5_sa_up'] are used for multiscale fusion
        net['concat_sa'] = torch.cat((net['out1_sa'], net['out2_sa_up'], net['out3_sa_up'], net['out4_sa_up'], net['out5_sa_up']), 1)
        net['concat_2ch'] = torch.cat((net['out1_2ch'], net['out2_2ch_up'], net['out3_2ch_up'], net['out4_2ch_up'], net['out5_2ch_up']), 1)
        net['concat_4ch'] = torch.cat((net['out1_4ch'], net['out2_4ch_up'], net['out3_4ch_up'], net['out4_4ch_up'], net['out5_4ch_up']), 1)

        net['concat'] = torch.cat((net['concat_sa'], net['concat_2ch'], net['concat_4ch']),1)
        net['comb_1'] = self.conv6(net['concat'])
        net['comb_2'] = self.conv7(net['comb_1'])

        net['conv3d0f'] = net['comb_2'].unsqueeze(1)

        net['conv3d_1'] = self.conv3d9(net['conv3d0f'])
        net['conv3d_2'] = self.conv3d10_1(self.conv3d10(net['conv3d_1']))
        net['conv3d_3'] = self.conv3d11_1(self.conv3d11(net['conv3d_2']))
        net['conv3d_4'] = self.conv3d12_1(self.conv3d12(net['conv3d_3']))
        net['conv3d_5'] = self.conv3d13_1(self.conv3d13(net['conv3d_4']))
        net['conv3d_6'] = self.conv3d14_1(self.conv3d14(net['conv3d_5']))
        net['conv3d_7'] = self.conv3d15_1(self.conv3d15(net['conv3d_6']))
        net['out'] = torch.tanh(self.conv3d16(net['conv3d_7']))

        # Estimate mesh
        net['concat_edge_ed'] = torch.cat((net['conv5_sa_ed'], net['conv5_2ch_ed'], net['conv5_4ch_ed']), 1)

        net['conv2d_0_ed'] = self.conv2d9(net['concat_edge_ed'])
        net['conv2d_1_ed'] = self.conv2d10_1(self.conv2d10(net['conv2d_0_ed']))
        net['conv2d_2_ed'] = self.conv2d11_1(self.conv2d11(net['conv2d_1_ed']))
        net['conv2d_3_ed'] = self.conv2d12_1(self.conv2d12(net['conv2d_2_ed']))
        net['conv2d_4_ed'] = self.conv2d13_1(self.conv2d13(net['conv2d_3_ed']))
        net['conv3d_edge_ed'] = net['conv2d_4_ed'].unsqueeze(1)
        net['out_edge_ed'] = self.conv3d18(self.conv3d17(net['conv3d_edge_ed']))

        net['concat_edge'] = torch.cat((net['conv5_sa'], net['conv5_2ch'], net['conv5_4ch']), 1)

        net['conv2d_0'] = self.conv2d9(net['concat_edge'])
        net['conv2d_1'] = self.conv2d10_1(self.conv2d10(net['conv2d_0']))
        net['conv2d_2'] = self.conv2d11_1(self.conv2d11(net['conv2d_1']))
        net['conv2d_3'] = self.conv2d12_1(self.conv2d12(net['conv2d_2']))
        net['conv2d_4'] = self.conv2d13_1(self.conv2d13(net['conv2d_3']))
        net['conv3d_edge'] = net['conv2d_4'].unsqueeze(1)
        net['out_edge'] = self.conv3d18(self.conv3d17(net['conv3d_edge']))


        return net


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


