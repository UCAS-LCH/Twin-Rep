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


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=20, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.0078, type=float,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    help='model for white-box attack evaluation')
parser.add_argument("--dataset", type=str, choices=["CIFAR10", "SVHN", "CIFAR100"], default="CIFAR10")
parser.add_argument("--model", type=str, choices=["ResNet18", "ResNet34", "ResNet50", "vgg16", 'WideResNet'], default="vgg16")

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



def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.epsilon * 2.5 /(args.num_steps+1)
                  #step_size=args.step_size
                  ):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).sum().item()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).sum().item()
    #print('err pgd (white-box): ', err_pgd)
    return err, err_pgd

def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_acc: ', 1 - natural_err_total/len(test_loader.dataset))
    print('robust_acc: ', 1 - robust_err_total/len(test_loader.dataset))

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
    eval_adv_test_whitebox(model, device, test_loader)


if __name__ == '__main__':
    main()
