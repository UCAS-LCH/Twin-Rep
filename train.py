from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

import models
from model_utils import *
from datasets import *
from attack import pgd_attack
from std_adv import adv_loss
from trades import trades_loss
from semisup import get_semisup_dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--score-decay', '--sd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.0078,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=123, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='./checkpoints',
                    help='directory of model for saving checkpoint')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument("--layer-type", type=str, choices=["dense", "twin", "channel", "mask"], default="twin")
parser.add_argument("--pruning-type", type=str, choices=["structure", "unstructure"], default="unstructure")
parser.add_argument("--pruning", type=float, default="0.0")
parser.add_argument("--mode", type=str, choices=['nat', 'adv', 'trades'], default="nat")
parser.add_argument('--dataset', choices=['CIFAR10', 'SVHN', 'CIFAR100', 'tiny_imagenet', 'imagenet'], default="CIFAR10")
parser.add_argument('--model', choices=['ResNet18', 'ResNet34', 'ResNet50', 'vgg16', 'WideResNet'], default="ResNet18")
parser.add_argument('--semisup', action='store_true', default=False,
                    help='whether to use tinyimage dataset')
parser.add_argument('--pruning-epoch', type=int, default=10, metavar='N',
                    help='when to process pruning')


args = parser.parse_args()
if args.semisup:
    datamode = "rst"
else:
    datamode = "none"


# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

# setup data loader
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

if args.dataset == 'CIFAR10' and args.semisup:
    sm_loader = get_semisup_dataloader(args.batch_size)
    print('load tinyimage dataset')
else:
    sm_loader = None


def train(args, model, device, train_loader, optimizer, epoch, lr):
    model.train()
    dataloader = train_loader if sm_loader is None else zip(train_loader, sm_loader)
    for batch_idx, data in enumerate(dataloader):
        if sm_loader:
            image, target = (
                torch.cat([d[0] for d in data], 0).to(device),
                torch.cat([d[1] for d in data], 0).to(device),
            )
        else:
            image, target = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        # calculate robust loss
        if args.mode == 'adv':
            loss = adv_loss(model=model,
                            x_natural=image,
                            y=target,
                            optimizer=optimizer,
                            step_size=args.step_size,
                            epsilon=args.epsilon,
                            perturb_steps=args.num_steps,
                            beta=args.beta)
        elif args.mode == 'trades':
            loss = trades_loss(model=model,
                               x_natural=image,
                               y=target,
                               optimizer=optimizer,
                               step_size=args.step_size,
                               epsilon=args.epsilon,
                               perturb_steps=args.num_steps,
                               beta=args.beta)
        else:
            loss = F.cross_entropy(model(image), target)
        loss.backward()
        #show_gradients_norm(model)

        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    train_loss_pgd = 0
    correct = 0
    correct_pgd = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            loss, loss_pgd, acc, acc_pgd = pgd_attack(model, data, target)
            correct += acc
            correct_pgd += acc_pgd
            train_loss += loss
            train_loss_pgd += loss_pgd
    train_loss /= len(train_loader.dataset)
    train_loss_pgd /= len(train_loader.dataset)
    train_accuracy = correct / len(train_loader.dataset)
    train_accuracy_pgd = correct_pgd / len(train_loader.dataset)
    print('train: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    print('train: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss_pgd, correct_pgd, len(train_loader.dataset),
        100. * correct_pgd / len(train_loader.dataset)))
    return train_loss, train_loss_pgd, train_accuracy, train_accuracy_pgd

def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_loss_pgd = 0
    correct = 0
    correct_pgd = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            loss, loss_pgd, acc, acc_pgd = pgd_attack(model, data, target)
            correct += acc
            correct_pgd += acc_pgd
            test_loss += loss
            test_loss_pgd += loss_pgd
    test_loss /= len(test_loader.dataset)
    test_loss_pgd /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    test_accuracy_pgd = correct_pgd / len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss_pgd, correct_pgd, len(test_loader.dataset),
        100. * correct_pgd / len(test_loader.dataset)))
    return test_loss, test_loss_pgd, test_accuracy, test_accuracy_pgd


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= args.epochs * 0.7:
        lr = args.lr * 0.1
    if epoch >= args.epochs * 0.85:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def unstructured_pruning(model, pruning_ratio, layer_type):
    weight_lists=[]
    if layer_type == "channel":
        for name, module in model.named_modules():
            if hasattr(module, 'mask') and module.weight.ndimension()==4:
                weight_lists.append(torch.einsum('pq,qrst->prst', [module.aux_weight.data, module.weight.data]).view(-1).cpu())
            elif hasattr(module, 'mask') and module.weight.ndimension()==2:
                weight_lists.append(torch.einsum('pq,qr->pr', [module.aux_weight.data, module.weight.data]).view(-1).cpu())
    elif layer_type == "mask":
        for name, module in model.named_modules():
            if hasattr(module, 'mask'):
                weight_lists.append(module.weight.data.view(-1).cpu())
    elif layer_type == "twin":
        for name, module in model.named_modules():
            if hasattr(module, 'mask'):
                weight_lists.append((module.aux_weight.data * module.weight.data).view(-1).cpu())
    else:
        raise ValueError("Incorrect layer type")

    all_weights=torch.cat(weight_lists)
    thres_index = round(all_weights.shape[0] * pruning_ratio)+1
    thres = all_weights.abs().kthvalue(thres_index).values.item()

    if layer_type == "channel":
        for name, module in model.named_modules():
            if hasattr(module, 'aux_weight') and module.weight.ndimension()==4:
                module.mask = torch.where(
                    torch.einsum('pq,qrst->prst', [module.aux_weight.data, module.weight.data]).abs().cpu() < thres,
                    torch.zeros(module.mask.size()), torch.ones(module.mask.size())).to(device)
            elif hasattr(module, 'aux_weight') and module.weight.ndimension()==2:
                module.mask = torch.where(
                    torch.einsum('pq,qr->pr', [module.aux_weight.data, module.weight.data]).abs().cpu() < thres,
                    torch.zeros(module.mask.size()), torch.ones(module.mask.size())).to(device)
    elif layer_type == "mask":
        for name, module in model.named_modules():
            if hasattr(module, 'mask'):
                module.mask = torch.where(
                    module.weight.data.abs().cpu() < thres,
                    torch.zeros(module.mask.size()), torch.ones(module.mask.size())).to(device)
    elif layer_type == "twin":
        for name, module in model.named_modules():
            if hasattr(module, 'aux_weight'):
                module.mask = torch.where(
                    (module.aux_weight.data * module.weight.data).abs().cpu() < thres,
                    torch.zeros(module.mask.size()), torch.ones(module.mask.size())).to(device)
    print(thres_index, thres)

def group_lasso(weight):
    lasso_vector = torch.sqrt(torch.sum(weight**2, dim=(1,2,3))).cpu()
    return lasso_vector

def structured_pruning(model, pruning_ratio, layer_type):
    metric_lists=[]
    if layer_type == "mask":
        for name, module in model.named_modules():
            if hasattr(module, 'mask') and module.weight.ndimension()==4:
                metric_lists.append(group_lasso(module.weight.data))
    elif layer_type == "twin":
        for name, module in model.named_modules():
            if hasattr(module, 'aux_weight') and module.weight.ndimension()==4:
                metric_lists.append(group_lasso(module.weight.data * module.aux_weight.data))
    elif layer_type == "channel":
        for name, module in model.named_modules():
            if hasattr(module, 'aux_weight') and module.weight.ndimension()==4:
                metric_lists.append(group_lasso(torch.einsum('pq,qrst->prst', [module.aux_weight.data, module.weight.data])))

    all_weights=torch.cat(metric_lists)
    thres_index = round(all_weights.shape[0] * pruning_ratio)+1
    thres = all_weights.abs().kthvalue(thres_index).values.item()
    for name, module in model.named_modules():
        if hasattr(module, 'aux_weight') and module.weight.ndimension()==4:
            if layer_type == "channel":
                metric_vector = group_lasso(torch.einsum('pq,qrst->prst', [module.aux_weight.data, module.weight.data]))
            elif layer_type == "twin":
                metric_vector = group_lasso(module.weight.data * module.aux_weight.data)
            elif layer_type == "mask":
                metric_vector = group_lasso(module.weight.data)

            num_remove = torch.sum(metric_vector < thres).item()
            num_remove = min(num_remove, module.aux_weight.data.shape[0]-1)
            indices = torch.sort(metric_vector).indices
            zero_indices = indices[:num_remove]
            module.aux_weight.data[zero_indices] = 0.0
    print(thres)

def main():
    cl, ll = get_layers(args.layer_type)
    model = models.__dict__[args.model](conv_layer=cl, linear_layer=ll, num_classes=num_class).to(device)
    #model = nn.DataParallel(model)
    score_params = [v for n, v in model.named_parameters() if 'popup_score' in n and v.requires_grad]
    weight_params = [v for n, v in model.named_parameters() if 'popup_score' not in n and v.requires_grad]
    optimizer = torch.optim.SGD(
        [
            {
                "params": score_params,
                "weight_decay": args.score_decay
            },
            {"params": weight_params, "weight_decay": args.weight_decay},
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    best_acc = 0
    best_acc_pgd = 0
    saved_model = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pt'.format(args.model, args.dataset, args.score_decay, args.weight_decay, args.layer_type, args.mode, args.pruning, args.pruning_epoch, datamode, args.pruning_type)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        current_lr = adjust_learning_rate(optimizer, epoch)

        # adversarial training

        if args.pruning > 0:
            if epoch == args.pruning_epoch:
                best_acc = 0
                best_acc_pgd = 0
                if args.pruning_type == 'structure':
                    structured_pruning(model, args.pruning, args.layer_type)
                else:
                    unstructured_pruning(model, args.pruning, args.layer_type)

        train(args, model, device, train_loader, optimizer, epoch, current_lr)

        # evaluation on natural examples
        print('================================================================')
        #train_loss, train_loss_pgd, train_acc, train_acc_pgd = eval_train(model, device, train_loader)
        test_loss, test_loss_pgd, test_acc, test_acc_pgd = eval_test(model, device, test_loader)
        print('================================================================')
        # save checkpoint
        if test_acc > best_acc:
            dense_state_dict = subnet_to_dense(model.state_dict(), args.layer_type)
            torch.save(dense_state_dict, os.path.join(model_dir, saved_model))
            best_acc = test_acc

        if test_acc_pgd > best_acc_pgd:
            dense_state_dict = subnet_to_dense(model.state_dict(), args.layer_type)
            torch.save(dense_state_dict, os.path.join(model_dir, 'pgd_'+saved_model))
            best_acc_pgd = test_acc_pgd

    dense_state_dict = subnet_to_dense(model.state_dict(), args.layer_type)
    torch.save(dense_state_dict, os.path.join(model_dir, 'last_'+saved_model))

if __name__ == '__main__':
    main()
