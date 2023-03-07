import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, conv_layer, linear_layer, bias=False, num_classes=10):
        super(VGG, self).__init__()
        self._bias_exist = bias
        self.conv_layer = conv_layer
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = linear_layer(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [self.conv_layer(in_channels, x, kernel_size=3, padding=1, bias=self._bias_exist),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=False)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



def vgg11(conv_layer, linear_layer, bias=False, num_classes=10):
    return VGG('VGG11', conv_layer, linear_layer, bias=bias, num_classes=num_classes)

def vgg13(conv_layer, linear_layer, bias=False, num_classes=10):
    return VGG('VGG13', conv_layer, linear_layer, bias=bias, num_classes=num_classes)

def vgg16(conv_layer, linear_layer, bias=False, num_classes=10):
    return VGG('VGG16', conv_layer, linear_layer, bias=bias, num_classes=num_classes)

def vgg19(conv_layer, linear_layer, bias=False, num_classes=10):
    return VGG('VGG19', conv_layer, linear_layer, bias=bias, num_classes=num_classes)
