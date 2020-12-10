"""Original implementation found here: https://github.com/meliketoy/wide-resnet.pytorch (WideResNet = WRN)
Pytorch Implementation of Sergey Zagoruyko's Wide Residual Networks (https://arxiv.org/pdf/1605.07146v2.pdf)

Look at:
    WRN28_2: create WRN model with 28 depth and 2 width

    WRN28_10: create WRN model with 28 depth and 10 width
        Based on the sourcepage, if trained using all dataset in a supervised way:
            - CIFAR10 should yield 96.21% top-1 accuracy
            - CIFAR100 should yield 81.02% top-1 accuracy and 95.41% top-5 accuracy
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys, os
import numpy as np
import random

act = torch.nn.LeakyReLU()

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes,dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(act(self.bn1(x))))
        out = self.conv2(act(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):

    def __init__(self, depth, widen_factor,dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet_v2 depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks,dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = act(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def WRN28_10(num_classes=10, dropout = 0.0):
    """Returns WideResNet model with 28 depth and 10 width

    Args:
        num_classes (int, optional): Defaults to 10 - Should be set to 100 for CIFAR100 or 1000 for Imagenet[-ILSVRC2012].
        dropout (float, optional): Defaults to 0.0.

    Returns:
        model
    """
    model = Wide_ResNet(depth=28, widen_factor=10, dropout_rate = dropout, num_classes=num_classes)
    return model

def WRN28_2(num_classes=10, dropout = 0.0):
    """Returns WideResNet model with 28 depth and 2 width

    Args:
        num_classes (int, optional): Defaults to 10 - Should be set to 100 for CIFAR100 or 1000 for Imagenet[-ILSVRC2012].
        dropout (float, optional): Defaults to 0.0.

    Returns:
        model
    """
    model = Wide_ResNet(depth =28, widen_factor =2,dropout_rate = dropout, num_classes = num_classes)
    return model