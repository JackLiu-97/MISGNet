import math
import torch
from torch import nn
from .pvtv2 import pvt_v2_b1
import torch.nn.functional as F
from models.mynet.MGL.mGL import MGLNet


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D

        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        try:
            pe = pe.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class decode(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, out_channel, norm_layer=nn.BatchNorm2d):
        super(decode, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_down, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channel * 2, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(out_channel)

    def forward(self, left, down):
        down_mask = self.conv_d1(down)
        left_mask = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(left_mask * down_, inplace=True)
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_mask + left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MyNet(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False):
        super(MyNet, self).__init__()

        self.decode1 = decode(128, 128, 128)
        self.decode2 = decode(64, 64, 64)
        self.decode3 = decode(64, 64, 64)
        self.backbone = pvt_v2_b1()  # [64, 128, 320, 512]
        path = "D:\Download\pvt_v2_b1.pth"
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.Translayer1 = BasicConv2d(320, 128, 1)
        self.Translayer2 = BasicConv2d(128, 64, 1)
        self.Translayer3 = BasicConv2d(128, 64, 1)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.MGLNet0 = MGLNet(num_clusters=2, dim=64)
        self.MGLNet1 = MGLNet(num_clusters=4, dim=128)
        self.MGLNet2 = MGLNet(num_clusters=8, dim=320)

        self.final2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, 2, 3, bn=False, relu=False)
        )

    def forward(self, imgs1, imgs2, labels=None):
        pvt = self.backbone(imgs1)
        pvt_img2 = self.backbone(imgs2)

        x1_1, x2_1 = self.MGLNet0(pvt[0], pvt_img2[0])
        x1_2, x2_2 = self.MGLNet1(pvt[1], pvt_img2[1])
        x1_3, x2_3 = self.MGLNet2(pvt[2], pvt_img2[2])
        out1 = self.decode1(torch.abs(x1_2 - x2_2), self.Translayer1(torch.abs(x1_3 - x2_3)))
        out2 = self.decode2(torch.abs(x1_1 - x2_1), self.Translayer2(torch.abs(x1_2 - x2_2)))
        out3 = self.decode3(out2, self.Translayer3(out1))
        out3 = self.upsample(self.final2(out3))
        return out3
