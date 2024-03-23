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

from data import get_dataset, get_dataloader, get_unlearn_loader, get_forget_remain_loader, split_class_data
from backbone import get_model
from method import run_method
from trainer import train_and_save
from eval import evaluate_summary
from util import report_sample_by_class
import BayesianLayers
from torch.utils.data import ConcatDataset, Subset


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
    trainset, testset, trainset_test, trainset_two_transform, num_classes = get_dataset(args)
    forget_index, remain_index = split_class_data(trainset, args.class_idx, args.class_unlearn)
    remain_index = np.random.choice(np.array(remain_index), len(forget_index)).tolist()
    binary_train_index = sorted(forget_index + remain_index)

    test_forget_index, test_remain_index = split_class_data(testset, args.class_idx, math.inf)
    test_remain_index = np.random.choice(np.array(test_remain_index), len(test_forget_index)).tolist()

    binary_train_set =  ConcatDataset([Subset(trainset_test, binary_train_index), Subset(trainset_two_transform, binary_train_index)])

    binary_train_loader = DataLoader(dataset=binary_train_set, batch_size=args.batch_size, shuffle=True)
    print(f"*** Dataloader successfully implemented: train:{len(binary_train_set)} ***")
    print(f"*** Dataset:{args.data_name} ***")

    model = torch.load(args.model_path, map_location='cpu').to(args.device) #should get_embeddings == True [1]
    for param in model.parameters():
        param.requires_grad = False
    
    loss_fn = nn.BCELoss(reduction='none')
    loss_sample = nn.BCELoss()
    assert isinstance(args.probe_path, list) 
    for mod_path in tqdm(args.probe_path):
        probe = torch.load(mod_path, map_location='cpu').to(args.device)
        train_xent_forget = train_xent_remain = 0
        sample = 0 

        probe.eval()
        for idx, (inputs, labels) in enumerate(tqdm(binary_train_loader)):
            inputs, labels = inputs.to(args.device), torch.where(labels == args.class_idx, 0, 1).to(args.device)
            labels = F.one_hot(labels, num_classes=2).float()
            with torch.no_grad():
                _ , embeddings = model(inputs, get_embeddings=True)
            logits = probe(embeddings)
            fg_idx, rm_idx = torch.where(labels[:,0]==1)[0], torch.where(labels[:,0]==0)[0]
            breakpoint() 
            result = torch.sum(loss_fn(logits, labels).detach() * labels, dim=1)
            total_sum = torch.sum(result)
            fg_sum = torch.sum(result[fg_idx])
            rm_sum = total_sum - fg_sum

            train_xent_forget += fg_sum.item()
            train_xent_remain += rm_sum.item()
            sample += loss_sample(logits, labels).detach().cpu().numpy() * inputs.shape[0]
        
    print("***********************************************************")
    print(f"forget data: {round(train_xent_forget)} | remain data: {round(train_xent_remain)} | total: {round(train_xent_forget + train_xent_remain)} | sample: {round(sample)}")
    print("***********************************************************")

if __name__ == '__main__':
    args = arg_parse()
    args.model_path = './checkcheck/ResNet18_cifar10_adam_seed6_retrain.pth'
    args.probe_path = ['./metric/probe_cifar10_adam_seed6.pth']
    seed_torch(0)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
