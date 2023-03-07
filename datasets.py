import torch
import torchvision.transforms as transforms
import os
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100, SVHN
from torch.utils.data import Subset, DataLoader


def imagenet_dataloader(batch_size=64, data_dir='../data/imagenet/'):
    train_set = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    test_set = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_class = 1000

    return train_loader, test_loader, num_class


def tiny_imagenet_dataloader(batch_size=64, data_dir='../data/tiny-imagenet-200/'):
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_path = os.path.join(data_dir, 'train/')
    test_path = os.path.join(data_dir, 'test/')


    train_set = ImageFolder(train_path, transform=train_transform)
    test_set = ImageFolder(test_path, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_class = 200

    return train_loader, test_loader, num_class

def cifar10_dataloader(batch_size=64, data_dir = '../data'):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_class = 10

    return train_loader, test_loader, num_class

def cifar100_dataloader(batch_size=64, data_dir = '../data'):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_class = 100

    return train_loader, test_loader, num_class

def svhn_dataloader(batch_size=64, data_dir = '../data'):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = SVHN(data_dir, split='train', transform=train_transform, download=True)
    test_set = SVHN(data_dir, split='test', transform=train_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_class = 10

    return train_loader, test_loader, num_class
