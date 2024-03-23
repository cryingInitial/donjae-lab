import os
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from util import TwoTransform

from torchvision import transforms


from torchvision.datasets import ImageFolder

def get_dataset(args):
    mean, std, image_size, num_classes = get_statistics(args.data_name)
    args.mean, args.std, args.image_size, args.num_classes = mean, std, image_size, num_classes
    
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    contastive_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    two_transform = TwoTransform(transform=contastive_transform, n_views=2)

    # if no dataset execute shell file
    if not os.path.exists(f'./dataset/{args.data_name}'):
        subprocess.run([f'./dataset/{args.data_name}.sh'])
        subprocess.run(['mv', f'./{args.data_name}', './dataset'])
        subprocess.run(['rm', f'./{args.data_name}_png.tar'])
    
    trainset = ImageFolder(root=f'./dataset/{args.data_name}/train', transform=train_transform)
    testset = ImageFolder(root=f'./dataset/{args.data_name}/test', transform=test_transform)
    trainset_test = ImageFolder(root=f'./dataset/{args.data_name}/train', transform=test_transform)

    trainset_two_transform = ImageFolder(root=f'./dataset/{args.data_name}/train', transform=contastive_transform)
    
    return trainset, testset, trainset_test, trainset_two_transform, num_classes


def get_dataloader(trainset, testset, trainset_test, args):
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)
    train_test_loader = DataLoader(dataset=trainset_test, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader, train_test_loader

def get_statistics(data_name):

    # CIFAR datasets
    if data_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        image_size = 32
        num_classes = 10
    elif data_name == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        image_size = 32
        num_classes = 100
    elif data_name == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image_size = 224
        num_classes = 1000
    # FGVC datasets
    elif data_name == 'cars':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image_size = 224
        num_classes = 196
    elif data_name == 'flowers':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image_size = 224
        num_classes = 102
    else: raise ValueError('Invalid dataset name')

    # if args.model_name == 'vit':
    #     processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    #     mean = processor.image_mean
    #     std = processor.image_std
    #     image_size = 224

    return mean, std, image_size, num_classes

def get_forget_remain_loader(set, args):
    if args.test_mode == 'class': 
        forget_index, remain_index = split_class_data(set, args.class_idx, args.class_unlearn)
        forget_set = torch.utils.data.Subset(set, forget_index)
        remain_set = torch.utils.data.Subset(set, remain_index)
        forget_loader = DataLoader(dataset=forget_set, batch_size=args.batch_size, shuffle=True)
        remain_loader = DataLoader(dataset=remain_set, batch_size=args.batch_size, shuffle=True)
    
    elif args.test_mode == 'sample':
        forget_index, remain_index = split_sample_data(set, args.sample_unlearn_per_class, args.data_name)
        forget_set = torch.utils.data.Subset(set, forget_index)
        remain_set = torch.utils.data.Subset(set, remain_index)
        forget_loader = DataLoader(dataset=forget_set, batch_size=args.batch_size, shuffle=True)
        remain_loader = DataLoader(dataset=remain_set, batch_size=args.batch_size, shuffle=True)
    
    return forget_set, remain_set, forget_loader, remain_loader


def get_unlearn_loader(trainset, testset, trainset_test, args):
    if args.test_mode == 'class': 
        forget_index, remain_index = split_class_data(trainset, args.class_idx, args.class_unlearn)
        test_forget_index, test_remain_index = split_class_data(testset, args.class_idx, math.inf)
        test_forget_set = torch.utils.data.Subset(testset, test_forget_index)
        test_remain_set = torch.utils.data.Subset(testset, test_remain_index)
        test_forget_loader = DataLoader(dataset=test_forget_set, batch_size=args.batch_size, shuffle=False)
        test_remain_loader = DataLoader(dataset=test_remain_set, batch_size=args.batch_size, shuffle=False)

    elif args.test_mode == 'sample': 
        forget_index, remain_index = split_sample_data(trainset, args.sample_unlearn_per_class, args.data_name)
        test_forget_set = None
        test_remain_set = testset
        test_forget_loader = None
        test_remain_loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)

    train_forget_set = torch.utils.data.Subset(trainset, forget_index)
    train_remain_set = torch.utils.data.Subset(trainset, remain_index)
    train_forget_loader = DataLoader(dataset=train_forget_set, batch_size=args.batch_size, shuffle=True)
    train_remain_loader = DataLoader(dataset=train_remain_set, batch_size=args.batch_size, shuffle=True)

    train_forget_test_set = torch.utils.data.Subset(trainset_test, forget_index)
    train_remain_test_set = torch.utils.data.Subset(trainset_test, remain_index)
    train_forget_test_loader = DataLoader(dataset=train_forget_test_set, batch_size=args.batch_size, shuffle=True)
    train_remain_test_loader = DataLoader(dataset=train_remain_test_set, batch_size=args.batch_size, shuffle=True)

    return train_forget_set, train_remain_set, test_forget_set, test_remain_set, train_forget_test_set, train_remain_test_set, \
        train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, train_forget_test_loader, train_remain_test_loader

def split_class_data(dataset, forget_class, num_forget):
    forget_index = []
    remain_index = []
    sum = 0
    for i, (data, target) in enumerate(dataset):
        if target == forget_class and sum < num_forget:
            forget_index.append(i)
            sum += 1
        else:
            remain_index.append(i)
    return forget_index, remain_index

def split_sample_data(dataset, num_forget_per_class, data_name):
    forget_index = []
    remain_index = []
    sum = [0 for _ in range(get_statistics(data_name)[-1])]
    for i, (data, target) in enumerate(dataset):
        if sum[target] < num_forget_per_class:
            forget_index.append(i)
            sum[target] += 1
        else:
            remain_index.append(i)

    return forget_index, remain_index