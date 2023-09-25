# src: https://github.com/FedML-AI/FedML/blob/ecd2d81222301d315ca3a84be5a5ce4f33d6181c/python/fedml/model/cv/resnet_gn.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

from models.group_normalization import GroupNorm2d

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def norm2d(planes, num_channels_per_group=32):
    #print("num_channels_per_group:{}".format(num_channels_per_group))
    if num_channels_per_group > 0:
        return GroupNorm2d(
            planes, num_channels_per_group, affine=True, track_running_stats=False
        )
    else:
        return nn.BatchNorm2d(planes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, group_norm=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm2d(planes, group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm2d(planes, group_norm)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, group_norm=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm2d(planes, group_norm)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = norm2d(planes, group_norm)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm2d(planes * 4, group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, group_norm=0):
        self.inplanes = 64
        super(ResNet, self).__init__()
        
        self.global_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.local_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.bn1 = norm2d(64, group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.global_layer1 = self._make_layer(block, 64, layers[0], group_norm=group_norm)
        self.global_layer2 = self._make_layer(block, 128, layers[1], stride=2, group_norm=group_norm)
        self.global_layer3 = self._make_layer(block, 256, layers[2], stride=2, group_norm=group_norm)
        self.global_layer4 = self._make_layer(block, 512, layers[3], stride=2, group_norm=group_norm)

        self.inplanes = 64
        self.local_layer1 = self._make_layer(block, 64, layers[0], group_norm=group_norm)
        self.local_layer2 = self._make_layer(block, 128, layers[1], stride=2, group_norm=group_norm)
        self.local_layer3 = self._make_layer(block, 256, layers[2], stride=2, group_norm=group_norm)
        self.local_layer4 = self._make_layer(block, 512, layers[3], stride=2, group_norm=group_norm)
        
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AvgPool2d(1)
        self.global_fc = nn.Linear(512 * block.expansion, num_classes)
        self.local_fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, GroupNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, Bottleneck):
                m.bn3.weight.data.fill_(0)
            if isinstance(m, BasicBlock):
                m.bn2.weight.data.fill_(0)

    def _make_layer(self, block, planes, blocks, stride=1, group_norm=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm2d(planes * block.expansion, group_norm),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, group_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, group_norm=group_norm))

        return nn.Sequential(*layers)

    def forward(self, x, alpha):
        alpha = F.gumbel_softmax(alpha)
        A = torch.argmax(alpha, dim=1)

        global_x = self.global_conv1(x)
        global_x = self.bn1(global_x)
        global_x = self.relu(global_x)
        global_x = self.maxpool(global_x)

        local_x = self.local_conv1(x)
        local_x = self.bn1(local_x)
        local_x = self.relu(local_x)
        local_x = self.maxpool(local_x)

        x = A[0] * global_x + (1 - A[0]) * local_x

        global_x = self.global_layer1(x)
        local_x = self.local_layer1(x)
        
        x = A[1] * global_x + (1 - A[1]) * local_x

        global_x = self.global_layer2(x)
        local_x = self.local_layer2(x)
        
        x = A[2] * global_x + (1 - A[2]) * local_x

        global_x = self.global_layer3(x)
        local_x = self.local_layer3(x)

        x = A[3] * global_x + (1 - A[3]) * local_x

        global_x = self.global_layer4(x)
        local_x = self.local_layer4(x)
        
        x = A[4] * global_x + (1 - A[4]) * local_x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        global_x = self.global_fc(x)
        local_x = self.local_fc(x)

        x = A[5] * global_x + (1 - A[5]) * local_x
        
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model

def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model