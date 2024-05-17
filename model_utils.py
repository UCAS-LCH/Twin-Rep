import torch
import torch.nn as nn
import torchvision

import os
import math
import numpy as np
from models.layers import *

class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)

def freeze_vars(model, var_name):
    assert var_name in ["weight", "bias", "aux_weight"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d,)):
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False
                    if getattr(v, var_name).grad is not None:
                        getattr(v, var_name).grad = None


def unfreeze_vars(model, var_name):
    assert var_name in ["weight", "bias", "aux_weight"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True


def get_layers(layer_type):
    if layer_type == "dense":
        return nn.Conv2d, nn.Linear
    elif layer_type == "twin":
        return TwinConv, TwinLinear
    elif layer_type == "channel":
        return ChannelConv, ChannelLinear
    elif layer_type == "mask":
        return MaskConv, MaskLinear
    else:
        raise ValueError("Incorrect layer type")


def show_gradients_norm(model, p=1):
    for i, v in model.named_parameters():
        if v.requires_grad:
            print(f"variable = {i}, Gradient requires_grad = {torch.norm(v.grad, p=p)}")


def subnet_to_dense(subnet_dict, layer_type):
    dense = {}

    if layer_type == 'dense':
        dense = subnet_dict
        return dense
        
    # load dense variables
    for (k, v) in subnet_dict.items():
        if "aux_weight" not in k and "mask" not in k:
            dense[k] = v

    # update dense variables
    if layer_type == 'mask':
        for (k, v) in subnet_dict.items():
            if "mask" in k:
                dense[k.replace("mask", "weight")] = (
                        subnet_dict[k.replace("mask", "weight")] * v)
        return dense
        
    for (k, v) in subnet_dict.items():
        if "aux_weight" in k:
            if layer_type == 'twin':
                dense[k.replace("aux_weight", "weight")] = (
                    subnet_dict[k.replace("aux_weight", "weight")] * v * subnet_dict[k.replace("aux_weight", "mask")])
            elif layer_type == 'channel':
                if subnet_dict[k.replace("aux_weight", "weight")].ndimension()==4:
                    dense[k.replace("aux_weight", "weight")] = torch.einsum('pq,qrst->prst',
                        [v, subnet_dict[k.replace("aux_weight", "weight")]]) * subnet_dict[k.replace("aux_weight", "mask")]
                elif subnet_dict[k.replace("aux_weight", "weight")].ndimension()==2:
                    dense[k.replace("aux_weight", "weight")] = torch.einsum('pq,qr->pr',
                        [v, subnet_dict[k.replace("aux_weight", "weight")]]) * subnet_dict[k.replace("aux_weight", "mask")]
    return dense

