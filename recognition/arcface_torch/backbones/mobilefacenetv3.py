#!/usr/bin/env python
# encoding: utf-8

import math
import torch
from torch import nn
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

def mobilefacenetv3():
    return MobileFaceNetV3(feature_dim=128)

if __name__ == "__main__":
    from torchscope import scope
    net = mobilefacenetv3()
    scope(net, input_size=(3, 112, 112))
    x = torch.randn(2,3,112,112)
    y = net(x)
    print(y.size())