import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


class TwinConv(nn.Conv2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(TwinConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.aux_weight = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.aux_weight, a=math.sqrt(5))
        self.register_buffer('mask', torch.ones(self.weight.shape), persistent=True)
        self.w = 0

    def forward(self, x):

        self.w = self.weight * self.aux_weight * self.mask
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class TwinLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(TwinLinear, self).__init__(in_features, out_features, bias)
        self.aux_weight = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.aux_weight, a=math.sqrt(5))
        self.register_buffer('mask', torch.ones(self.weight.shape), persistent=True)
        self.w = 0

    def forward(self, x):

        self.w = self.weight * self.aux_weight * self.mask
        x = F.linear(x, self.w, self.bias)

        return x

class ChannelConv(nn.Conv2d):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(ChannelConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.aux_weight = Parameter(torch.Tensor(out_channels, out_channels))
        nn.init.kaiming_uniform_(self.aux_weight, a=math.sqrt(5))
        self.register_buffer('mask', torch.ones(self.weight.shape), persistent=True)
        #self.register_buffer('mask', torch.ones(out_channels), persistent=True)
        self.w = 0

    def forward(self, x):

        self.w = torch.einsum('pq,qrst->prst', [self.aux_weight, self.weight]) * self.mask
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class ChannelLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(ChannelLinear, self).__init__(in_features, out_features, bias)
        self.aux_weight = Parameter(torch.Tensor(out_features, out_features))
        nn.init.kaiming_uniform_(self.aux_weight, a=math.sqrt(5))
        self.register_buffer('mask', torch.ones(self.weight.shape), persistent=True)
        #self.register_buffer('mask', torch.ones(out_features), persistent=True)
        self.w = 0

    def forward(self, x):

        self.w = torch.einsum('pq,qr->pr', [self.aux_weight, self.weight]) * self.mask
        x = F.linear(x, self.w, self.bias)

        return x

class MaskConv(nn.Conv2d):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(MaskConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.register_buffer('mask', torch.ones(self.weight.shape), persistent=True)
        self.w = 0

    def forward(self, x):

        self.w = self.weight * self.mask
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class MaskLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(self.weight.shape), persistent=True)
        self.w = 0

    def forward(self, x):

        self.w = self.weight * self.mask
        x = F.linear(x, self.w, self.bias)




