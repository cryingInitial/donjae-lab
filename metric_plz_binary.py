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
import os
import argparse
import numpy as np
import torch
import logging.config
import json
from tqdm import tqdm
from adv_generator import inf_generator

from data import get_dataset, get_dataloader, get_unlearn_loader, get_forget_remain_loader, split_class_data, get_statistics 
from backbone import get_model
from method import run_method
from trainer import train_and_save
from eval import evaluate_summary
from util import report_sample_by_class
import BayesianLayers
from torch.utils.data import ConcatDataset, Subset, Dataset


class FilteredDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None, exclude=True):
        self.dataset = ImageFolder(root_dir, transform=transform)  # Apply transformations when loading images
        self.classes = classes
        self.exclude = exclude
        self.filtered_indices = self._filter_indices()
        self.transform = transform  # Store the transform for optional use in __getitem__

    def _filter_indices(self):
        # Efficiently filter indices based on metadata/annotations
        indices = []
        for idx, (_, class_idx) in enumerate(self.dataset.samples):
            if self.exclude:
                if class_idx not in self.classes:
                    indices.append(idx)
            else:
                if class_idx in self.classes:
                    indices.append(idx)

        return indices

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # Map the filtered index back to the dataset's original indexing
        original_idx = self.filtered_indices[idx]
        image, label = self.dataset[original_idx]

        return image, label



class Probe(nn.Module):

  def print_param_count(self):
    total_params = 0
    for param in self.parameters():
      total_params += np.prod(param.size())
    print('Probe has {} parameters'.format(total_params))

class BayesProbe(Probe):
    """ Computes an MLP function of pairs of vectors.

    For a batch of sentences, computes all n scores
    for each sentence in the batch.
    """

    def __init__(self, args):
        print('Constructing BayesProbe')
        super(BayesProbe, self).__init__()
        self.args = args

        self.model_dim = 512
        self.label_space_size = 2
        #------------------------------------------------
        # diff
        self.relu = nn.ReLU()
        # layers
        self.fc1 = BayesianLayers.LinearGroupNJ(self.model_dim, 128, clip_var=0.04, cuda=True)
        self.fc2 = BayesianLayers.LinearGroupNJ(128, self.label_space_size, cuda=True)
        # self.fc3 = BayesianLayers.LinearGroupNJ(64, self.label_space_size, cuda=True)
        self.sigmoid = nn.Sigmoid()
        # layers including kl_divergence
        # self.kl_list = [self.fc1]
        self.kl_list = [self.fc1, self.fc2]
        # self.kl_list = [self.fc1, self.fc2, self.fc3]

        # end diff    
        #------------------------------------------------
        self.to(args.device)
        self.print_param_count()

    def forward(self, x):
        x = x.view(-1, 512)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.fc3(x)
        x = self.sigmoid(x)
        return x

    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD
    
    def get_masks(self, thresholds, return_log_alpha=False):
        masks = []
        alphas = []
        mask = None
        for i, (layer, threshold) in enumerate(zip(self.kl_list, thresholds)):
            log_alpha = layer.get_log_dropout_rates().cpu().data.numpy()
            mask = log_alpha < thresholds[i]
            alphas.append(log_alpha)
            masks.append(mask)

        if return_log_alpha:
            return masks, alphas
        return masks


def initialize_probe(args):
    model = BayesProbe(args)
    return model


def eval(current_itr, model, probe, test_loader, args):
    correct = 0
    probe.eval()
    for idx, (imputs, labels) in enumerate(tqdm(test_loader)):
        inputs, labels = imputs.to(args.device), torch.where(labels == args.class_idx, 0, 1).to(args.device)
        # labels = F.one_hot(labels, num_classes=2).float()
        with torch.no_grad():
            _ , embeddings = model(inputs, get_embeddings=True)
        output = probe(embeddings)
        pred = output.data.max(1)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
        
    print(f"Test Acc: {((correct/len(test_loader.dataset))*100):.3f}% in itr: {current_itr}")

    return (correct/len(test_loader.dataset))*100 

def seed_torch(seed):
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def arg_parse():
    parser = argparse.ArgumentParser("Boundary Unlearning")
    parser.add_argument('--rnd_seed', type=int, default=0, help='random seed') 
    parser.add_argument('--data_name', type=str, default='cifar10', help='dataset, mnist or cifar10')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--note', type=str, default='gold', help='note for each tests')
    # class unlearning, test_mode=class
    parser.add_argument('--class_idx', type=int, default=4, help='class index to unlearn')
    parser.add_argument('--class_unlearn', type=int, default=5000, help='number of unlearning samples')

    # sample unlearning, test_mode=sample
    args = parser.parse_args()
    return args


def main(args):
    #Prepare Dataloader
    '''
    Preparing balanced train, test loader
    '''
    binary_train_set = binary_test_set = None
    binary_train_loader = Binary_test_loader = None
    N = 0


    # mean, std, image_size, num_classes = get_statistics(args.data_name, args)
    mean, std, image_size, num_classes = get_statistics(args.data_name)
    normalize = transforms.Normalize(mean=mean, std=std)
    contastive_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])
    train_transform_2 = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomRotation((-60,60)),
            transforms.ToTensor(),
            normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # num_transform = 20 
    # transform_container = []
    
    # for i in range(num_transform):
        

    #         transforms.Compose([
    #             transforms.RandomCrop(image_size, padding=4),
    #             transforms.RandomRotation((-30,30)),
    #             transforms.GaussianBlur(),
    #             t
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             normalize,
    #     ])


    if args.data_name == 'cifar10':
        trainset, testset, trainset_test, trainset_two_transform, num_classes = get_dataset(args)
        forget_index, remain_index = split_class_data(trainset, args.class_idx, args.class_unlearn)
        remain_index = np.random.choice(np.array(remain_index), len(forget_index)).tolist()
        binary_train_index = sorted(forget_index + remain_index)

        test_forget_index, test_remain_index = split_class_data(testset, args.class_idx, math.inf)
        test_remain_index = np.random.choice(np.array(test_remain_index), len(test_forget_index)).tolist()
        binary_test_index = sorted(test_forget_index + test_remain_index)
    
        binary_train_set =  ConcatDataset([Subset(trainset_test, binary_train_index), Subset(trainset_two_transform, binary_train_index)])
        binary_test_set = Subset(testset, binary_test_index) 
        N = len(binary_train_set)
    
        binary_train_loader = DataLoader(dataset=binary_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        binary_test_loader =  DataLoader(dataset=binary_test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    elif args.data_name == 'cifar100':

        trainset, testset, trainset_test, trainset_two_transform, num_classes = get_dataset(args)
        train_forget_set = ConcatDataset([FilteredDataset(f'./dataset/{args.data_name}/train', list(range(0, 5)), transform=train_transform, exclude=False), FilteredDataset(f'./dataset/{args.data_name}/train', list(range(0, 5)), transform=train_transform_2, exclude=False), FilteredDataset(f'./dataset/{args.data_name}/train', list(range(0, 5)), transform=contastive_transform, exclude=False)])
        train_remain_set = ConcatDataset([FilteredDataset(f'./dataset/{args.data_name}/train', list(range(0, 5)), transform=train_transform, exclude=True), FilteredDataset(f'./dataset/{args.data_name}/train', list(range(0, 5)), transform=train_transform_2, exclude=True), FilteredDataset(f'./dataset/{args.data_name}/train', list(range(0, 5)), transform=contastive_transform, exclude=True)])
        total_forget_length = len(train_forget_set)
        total_remain_length = len(train_remain_set)

        train_remain_idx = sorted(np.random.choice(np.arange(total_remain_length), total_forget_length).tolist())
        train_remain_set = Subset(train_remain_set, train_remain_idx)
        binary_train_set = ConcatDataset([train_forget_set, train_remain_set]) 
    
        test_forget_set = FilteredDataset(f'./dataset/{args.data_name}/test', list(range(0, 5)), transform=test_transform, exclude=False)
        test_remain_set =  FilteredDataset(f'./dataset/{args.data_name}/test', list(range(0, 5)), transform=test_transform, exclude=True)
        test_remain_idx = sorted(np.random.choice(np.arange(len(train_remain_set)), len(test_forget_set)).tolist())
        test_remain_set  = Subset(test_remain_set, test_remain_idx)
        binary_test_set  = ConcatDataset([test_forget_set, test_remain_set])
        
        N = len(binary_train_set)
        
        binary_train_loader = DataLoader(dataset=binary_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        binary_train_loader = DataLoader(dataset=binary_test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)


    else:
        raise NotImplementedError
        trainset, testset, trainset_test, trainset_two_transform, num_classes = get_dataset(args)
        train_forget_set = ConcatDataset([FilteredDataset(f'./dataset/{args.data_name}/train', list(range(205, 210)), transform=trainset.transform, exclude=False), FilteredDataset(f'./dataset/{args.data_name}/train', list(range(205, 210)), transform=contastive_transform, exclude=False)])
        train_remain_set = ConcatDataset([FilteredDataset(f'./dataset/{args.data_name}/train', list(range(205, 210)), transform=trainset.transform, exclude=True), FilteredDataset(f'./dataset/{args.data_name}/train', list(range(205, 210)), transform=contastive_transform, exclude=True)])
        train_remain_idx = sorted(np.random.choice(np.arange(len(train_remain_set.dataset)), len(train_forget_set.dataset)).tolist())
        train_remain_set = Subset(train_remain_set, train_remain_idx)
        binary_train_set = ConcatDataset([train_forget_set, train_remain_set]) 
    
        test_forget_set = FilteredDataset(f'./dataset/{args.data_name}/train', list(range(0, 5)), transform=testset.transform, exclude=False)
        test_remain_set =  FilteredDataset(f'./dataset/{args.data_name}/train', list(range(0, 5)), transform=testset.transform, exclude=True)

        test_remain_idx = sorted(np.random.choice(np.arange(len(test_remain_set.dataset)), len(test_forget_set.dataset)).tolist())
        test_remain_set  = Subset(test_remain_set, test_remain_idx)
        binary_test_set  = ConcatDataset([test_forget_set, test_remain_set])
        
        N = len(binary_train_set)

        binary_train_loader = DataLoader(dataset=binary_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        binary_train_loader = DataLoader(dataset=binary_test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)


    print(f"*** Dataloader successfully implemented: train:{len(binary_train_set)} test:{len(binary_test_set)} ***")
    print(f"*** Dataset:{args.data_name} ***")


    '''
    Preparing Model. Backbone and probe
    '''
    # model = torch.load(args.model_path, map_location='cpu').to(args.device) #should get_embeddings == True [1]
    model = get_model().to(args.device)
    for param in model.parameters():
        param.requires_grad = False
    
    # sample = model(torch.randn(52, 3, 32, 32).cuda(), get_embeddings=True)[1]
    # sample = sample.cuda()

    linear_probe = initialize_probe(args)
    print(f"*** Model, Probe successfully implemented ***")

    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                          mode='min',
                                                          factor=0.5,
                                                          patience=0) 
    max_iter = 0
    if args.data_name == 'cifar10':
        max_iter = 50000
    elif args.data_name == 'cifar100':
        max_iter = 40000
    else:
        max_iter = 50000

    data_gen = inf_generator(binary_train_loader)

    final_report = None
    #Train binary classification task with given loss
    for itr in tqdm(range(1, max_iter+1), desc='[Batch Training]'):
        x_train, y_train = data_gen.__next__()
        x_train, y_train = x_train.to(args.device), torch.where(y_train == args.class_idx, 0, 1).to(args.device)
        y_train = F.one_hot(y_train, num_classes=2).float()

        with torch.no_grad():
            _ , embeddings = model(x_train, get_embeddings=True)
        logits = linear_probe(embeddings)

        bce_loss = loss_fn(logits, y_train)
        total_loss = bce_loss + (linear_probe.kl_divergence() / N)

        # linear_probe.zero_grad()
        optimizer.zero_grad()
        total_loss.backward() 
        optimizer.step()

        for layer in linear_probe.kl_list:
                layer.clip_variances()


        if itr % 2000 == 0:
            print('*** Eval period, doing eval ***')
            # eval(itr, model, linear_probe, binary_test_loader, args)
            train_acc = eval(itr, model, linear_probe, binary_train_loader, args)
            # linear_probe.zero_grad()
            linear_probe.train()
            print(f'{float(linear_probe.kl_divergence().detach().cpu().numpy()):.2f} : KL divergence ')
            

        if itr == max_iter:
            print("*** train finished, storing result ***")
            kl_div = 0
            train_xent = 0
            linear_probe.eval()
            for idx, (inputs, labels) in enumerate(tqdm(binary_train_loader)):
                inputs, labels = inputs.to(args.device), torch.where(labels == args.class_idx, 0, 1).to(args.device)
                labels = F.one_hot(labels, num_classes=2).float()
                with torch.no_grad():
                    _ , embeddings = model(inputs, get_embeddings=True)
                logits = linear_probe(embeddings)
                bce_loss = loss_fn(logits, labels)
                train_xent += bce_loss.detach().cpu().numpy() * inputs.shape[0]
            
            kl_div = float(linear_probe.kl_divergence().detach().cpu().numpy())
            final_acc = eval(itr, model, linear_probe, binary_test_loader, args)
            final_train_acc = eval(itr, model, linear_probe, binary_train_loader, args)
            final_report = {
                    'train_xent': train_xent,
                    'kl': kl_div, 
                    'final_test_acc': final_acc,
                    'final_train_acc': final_train_acc
            }
             
            with open(f'./3_8_train_report_{args.note}.json', 'w') as f:
                json.dump(final_report, f)

            #save model
            torch.save(linear_probe, f'./metric/3_8_probe_{args.note}.pth')
            

    #Evaluate Result
    # uniform_codelength = N * np.log2(2)
    uniform_codelength = N 
    model_bits, data_bits = final_report['kl'], final_report['train_xent']
    final_accuracy, final_train_accuracy = final_report['final_test_acc'], final_report['final_train_acc']
    total_code = model_bits + data_bits

    print("***********************************************************")
    print(f"Final Test Acc: {final_accuracy:.3f}% | Final Train Acc: {final_train_accuracy:.3f}")
    print(f"Model code: {round(model_bits)}bits | Data code: {round(data_bits)}bits")
    print("Variational codelength: {} bits".format(round(total_code)))
    print("Compression: {} ".format(round(uniform_codelength / data_bits, 2)))
    print("***********************************************************")
    

if __name__ == '__main__':
    args = arg_parse()
    # args.model_path = './checkpoints/ResNet18_cifar100_ori.pth'
    # args.model_path = 'checkpoints/ResNet18_cifar10_class_4_5000_retrain.pth' 
    # args.model_path = './checkpoints/ResNet18_cifar100_class_4_500_retrain.pth'
    # args.model_path = './checkpoints/ft_cifar10_class_4_5000_1.0_0.pth'
    # args.model_path = './checkpoints/teacher_cifar10_class_4_5000_1.0_0.pth'
    # args.model_path = './checkpoints/distill_cifar10_class_4_5000_1.0_0.pth'
    # args.model_path = './checkpoints/distill_cifar10_class_4_5000_1.0_0_first_trial.pth'
    # args.model_path = './checkpoints/distill_cifar10_class_4_5000_1.0_0_second_trial.pth'
    # args.model_path = './checkpoints/ft_cifar10_class_4_5000_1.0_0_50000.pth'
    # args.model_path = './checkpoints/ft_cifar10_class_4_5000_1.0_0_20000.pth'
    # args.model_path = './checkpoints/scrub_cifar10_class_4_5000_1.0_2024_2000.pth'
    # args.model_path='./checkpoints/ResNet18_cifar100_ori.pth'
    # args.model_path='./checkpoints/ResNet18_cifar100_class_4_500_retrain.pth'
    # args.model_path='./checkpoints/ft_cifar100_class_4_500_1.0_2024_2000.pth'
    # args.model_path='./checkpoints/ResNet18_cifar100_ori.pth'
    # args.model_path = './checkpoints/distill_cifar10_class_4_5000_1.0_2024_first_trial_1500.pth'
    # args.model_path = './checkcheck/ResNet18_cifar10_adam_seed7_retrain.pth'
    # args.model_path = './checkpoints/distill_cifar100_class_4_500_1.0_2024_2000.pth'

    # args.model_path = './checkcheck100/ResNet18_cifar100_adam_seed3_retrain.pth'
    seed_torch(3024)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
