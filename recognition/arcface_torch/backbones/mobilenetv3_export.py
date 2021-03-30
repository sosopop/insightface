#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: mobilefacenet.py
@time: 2018/12/21 15:45
@desc: mobilefacenet backbone
'''

import torch
from torch import nn
import math

MobileFaceNet_BottleNeck_Setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

class BottleNeck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(BottleNeck, self).__init__()
        self.connect = stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # 1*1 conv
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # 3*3 depth wise conv
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # 1*1 conv
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)

        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


class MobileFaceNet(nn.Module):
    def __init__(self, feature_dim=128, bottleneck_setting=MobileFaceNet_BottleNeck_Setting):
        super(MobileFaceNet, self).__init__()
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.cur_channel = 64
        block = BottleNeck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(128, 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, 7, 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, feature_dim, 1, 1, 0, linear=True)
        # self.linear0 = nn.Linear(512, feature_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.cur_channel, c, s, t))
                else:
                    layers.append(block(self.cur_channel, c, 1, t))
                self.cur_channel = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        # x = self.linear0(x)

        return x



import torch.nn.functional as F
from torch.nn import init

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV3_Large(nn.Module):
    def __init__(self, feature_dim=128):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )

        self.conv2 = nn.Conv2d(160, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.hs2 = hswish()
        
        self.conv3 = nn.Conv2d(512, 512, kernel_size=7, stride=1, padding=0, groups=512, bias=False)
        self.bn3 = nn.BatchNorm2d(512)

        self.conv4 = nn.Conv2d(512, feature_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(feature_dim)
        # self.linear3 = nn.Linear(960, 1280)
        # self.bn3 = nn.BatchNorm1d(1280)
        # self.hs3 = hswish()
        # self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        # out = F.avg_pool2d(out, 7)
        out = self.bn3(self.conv3(out))
        out = self.bn4(self.conv4(out))
        out = out.view(out.size(0), -1)
        # out = self.hs3(self.bn3(self.linear3(out)))
        # out = self.linear4(out)
        return out


class MobileNetV3_Small(nn.Module):
    def __init__(self, feature_dim=128):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        #kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride
        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )

        # V1
        # MobileFaceNet_BottleNeck_Setting = [
        #     # t, c , n ,s
        #     [2, 64, 5, 2],
        #     [4, 128, 1, 2],
        #     [2, 128, 6, 1],
        #     [4, 128, 1, 2],
        #     [2, 128, 2, 1]
        # ]
        # inp, oup, stride, expansion
        # layers.append(block(self.cur_channel, c, s, t))
        # self.conv2 = ConvBlock(128, 512, 1, 1, 0)

        self.conv2 = nn.Conv2d(96, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.hs2 = hswish()

        self.conv3 = nn.Conv2d(512, 512, kernel_size=7, stride=1, padding=0, groups=512, bias=False)
        self.bn3 = nn.BatchNorm2d(512)

        self.conv4 = nn.Conv2d(512, feature_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(feature_dim)

        # self.linear3 = nn.Linear(512, feature_dim)
        # self.bn3 = nn.BatchNorm1d(feature_dim)
        # self.hs3 = hswish()
        # self.linear4 = nn.Linear(1280, num_classes)
        # self.linear7 = ConvBlock(512, 512, 7, 1, 0, dw=True, linear=True)
        # self.linear1 = ConvBlock(512, feature_dim, 1, 1, 0, linear=True)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.bn4(self.conv4(out))
        # out = F.avg_pool2d(out, 7)
        # out = self.linear7(out)
        # out = self.linear1(out)
        out = out.view(out.size(0), -1)
        #out = self.bn3(self.linear3(out))
        #out = self.hs3(self.bn3(self.linear3(out)))
        #out = self.linear4(out)
        return out

MobileFaceNetV3_BottleNeck_Setting = [
    # t, c , n ,s
    [2, 64, 5, 2, 3],
    [4, 128, 1, 2, 3],
    [2, 128, 6, 1, 5],
    [4, 128, 1, 2, 5],
    [2, 128, 2, 1, 5]
]

class MobileFaceNetV3(nn.Module):
    def __init__(self, feature_dim=128):
        super(MobileFaceNetV3, self).__init__()
        
        self.cur_channel = 64

        # self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        # self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.hs1 = hswish()

        
        self.dw_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.dw_bn1 = nn.BatchNorm2d(64)
        self.dw_hs1 = hswish()

        self.bneck = self._make_layer(Block, MobileFaceNetV3_BottleNeck_Setting)

        self.conv2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.hs2 = hswish()

        self.conv3 = nn.Conv2d(512, 512, kernel_size=7, stride=1, padding=0, groups=512, bias=False)
        self.bn3 = nn.BatchNorm2d(512)

        self.conv4 = nn.Conv2d(512, feature_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(feature_dim)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s, k in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(k, self.cur_channel, t * self.cur_channel, c, hswish(), SeModule(c), s))
                else:
                    layers.append(block(k, self.cur_channel, t * self.cur_channel, c, hswish(), SeModule(c), 1))
                self.cur_channel = c

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.dw_hs1(self.dw_bn1(self.dw_conv1(out)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.bn4(self.conv4(out))
        out = out.view(out.size(0), -1)
        return out



# class MobileFaceNet(nn.Module):
#     def __init__(self, feature_dim=128, bottleneck_setting=MobileFaceNet_BottleNeck_Setting):
#         super(MobileFaceNet, self).__init__()
#         self.conv1 = ConvBlock(3, 64, 3, 2, 1)
#         self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

#         self.cur_channel = 64
#         block = BottleNeck
#         self.blocks = self._make_layer(block, bottleneck_setting)

#         self.conv2 = ConvBlock(128, 512, 1, 1, 0)
#         self.linear7 = ConvBlock(512, 512, 7, 1, 0, dw=True, linear=True)
#         self.linear1 = ConvBlock(512, feature_dim, 1, 1, 0, linear=True)
#         # self.linear0 = nn.Linear(512, feature_dim)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(self, block, setting):
#         layers = []
#         for t, c, n, s in setting:
#             for i in range(n):
#                 if i == 0:
#                     layers.append(block(self.cur_channel, c, s, t))
#                 else:
#                     layers.append(block(self.cur_channel, c, 1, t))
#                 self.cur_channel = c

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.dw_conv1(x)
#         x = self.blocks(x)
#         x = self.conv2(x)
#         x = self.linear7(x)
#         x = self.linear1(x)
#         x = x.view(x.size(0), -1)

# def test():
#     from torchscope import scope
#     net = MobileNetV3_Small()
#     scope(net, input_size=(3, 112, 112))
#     x = torch.randn(2,3,112,112)
#     y = net(x)
#     print(y.size())

# test()

# if __name__ == "__main__":
#     input = torch.Tensor(2, 3, 112, 112)
#     net = MobileFaceNet()
#     print(net)

#     x = net(input)
#     print(x.shape)

def mbv3_export(pretrained=False, progress=True, **kwargs):
    # return MobileNetV3_Small(feature_dim=128)
    # return MobileNetV3_Large(feature_dim=128)
    # return MobileFaceNet(feature_dim=128)
    return MobileFaceNetV3(feature_dim=128)

# def test():
#     from torchscope import scope
#     net = mbv3()
#     scope(net, input_size=(3, 112, 112))
#     x = torch.randn(2,3,112,112)
#     y = net(x)
#     print(y.size())

# test()