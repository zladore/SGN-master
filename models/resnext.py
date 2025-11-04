import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import conv1x1x1, Bottleneck, ResNet
from utils import partialclass


def get_inplanes():
    return [128, 256, 512, 1024]


class ResNeXtBottleneck(Bottleneck):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super().__init__(inplanes, planes, stride, downsample)

        mid_planes = cardinality * planes // 32
        self.conv1 = conv1x1x1(inplanes, mid_planes)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes,
                               mid_planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               groups=cardinality,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = conv1x1x1(mid_planes, planes * self.expansion)


class ResNeXt(ResNet):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 cardinality=32,
                 n_classes=250):
        block = partialclass(block, cardinality=cardinality)
        super().__init__(block, layers, block_inplanes,
                         n_input_channels, conv1_t_size,
                         conv1_t_stride, no_max_pool,
                         shortcut_type, n_classes=n_classes)

        # 🔧 保留 y 方向，压缩其他方向
        self.avgpool = nn.AdaptiveAvgPool3d((1, 250, 1))
        # 最后得到 [B, C, 1, 250, 1]

        # 🔧 用 1x1 卷积映射到 1 通道（即每个 y 上一个预测值）
        self.fc = nn.Conv3d(1024 * ResNeXtBottleneck.expansion, 1, kernel_size=1)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.bn1(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        if not self.no_max_pool:
            x = self.maxpool(x)
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print("****")
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)

        x = self.avgpool(x)  # -> [B, C, 1, 250, 1]
        print(x.shape)
        x = self.fc(x)       # -> [B, 1, 1, 250, 1]
        print(x.shape)
        x = x.squeeze(1).squeeze(1).squeeze(-1)  # -> [B, 250]
        print(x.shape)
        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [50, 101, 152, 200]

    if model_depth == 50:
        model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], get_inplanes(),
                        **kwargs)
    elif model_depth == 101:
        model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], get_inplanes(),
                        **kwargs)
    elif model_depth == 152:
        model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], get_inplanes(),
                        **kwargs)
    elif model_depth == 200:
        model = ResNeXt(ResNeXtBottleneck, [3, 24, 36, 3], get_inplanes(),
                        **kwargs)

    return model