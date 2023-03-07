from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import models
from model_utils import *
from datasets import *
from collections import OrderedDict
from autoattack import AutoAttack

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='model for white-box attack evaluation')
parser.add_argument("--dataset", type=str, choices=["CIFAR10", "SVHN", "CIFAR100"], default="CIFAR10")
parser.add_argument("--model", type=str, choices=["ResNet18", "ResNet34", "ResNet50", "vgg16", 'WideResNet'], default="vgg16")
parser.add_argument('--log-path', type=str, default='./log_file.txt')
parser.add_argument('--version', type=str, default='standard')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# set up data loader
if args.dataset == 'CIFAR10':
    train_loader, test_loader, num_class = cifar10_dataloader(batch_size=args.batch_size)
    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
elif args.dataset == 'CIFAR100':
    train_loader, test_loader, num_class = cifar100_dataloader(batch_size=args.batch_size)
    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
elif args.dataset == 'SVHN':
    train_loader, test_loader, num_class = svhn_dataloader(batch_size=args.batch_size)
    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
elif args.dataset == 'tiny_imagenet':
    train_loader, test_loader, num_class = tiny_imagenet_dataloader(batch_size=args.batch_size)
    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
elif args.dataset == 'imagenet':
    train_loader, test_loader, num_class = imagenet_dataloader(batch_size=args.batch_size)
    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def main():

    cl, ll = get_layers('dense')
    model = models.__dict__[args.model](conv_layer=cl, linear_layer=ll, num_classes=num_class).to(device)
    state_dict = torch.load(args.model_path)

    new_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            new_k = k[7:]
        else:
            new_k = k
        new_dict[new_k] = v

    model.load_state_dict(new_dict)
    model.eval()

    adversary = AutoAttack(model, norm='Linf', eps=args.epsilon, log_path=args.log_path, version=args.version)

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    if args.version == 'custom':
        adversary.attacks_to_run = ['square']
        for n_queries in [50, 100, 250, 500, 1000]:
            adversary.square.n_queries = n_queries
            with torch.no_grad():
                adversary.run_standard_evaluation(x_test, y_test,
                    bs=args.test_batch_size)
    else:
        with torch.no_grad():
            adversary.run_standard_evaluation(x_test, y_test,
                bs=args.test_batch_size)


if __name__ == '__main__':
    main()
