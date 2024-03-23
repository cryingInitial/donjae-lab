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

        if args.data_name == 'cifar10':
            self.label_space_size = 10
        else:
            self.label_space_size = 2
        #------------------------------------------------
        # diff
        self.relu = nn.ReLU()
        # layers
        self.fc1 = BayesianLayers.LinearGroupNJ(self.model_dim, 128, clip_var=0.04, cuda=True)
        self.fc2 = BayesianLayers.LinearGroupNJ(128, self.label_space_size, cuda=True)
        # self.fc3 = BayesianLayers.LinearGroupNJ(64, self.label_space_size, cuda=True)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()
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
        # x = self.softmax(x)
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
        inputs, labels = imputs.to(args.device), labels.to(args.device)
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
    # remain_index = np.random.choice(np.array(remain_index), len(forget_index)).tolist()
    original_train_index = sorted(forget_index + remain_index)

    test_forget_index, test_remain_index = split_class_data(testset, args.class_idx, math.inf)
    # test_remain_index = np.random.choice(np.array(test_remain_index), len(test_forget_index)).tolist()
    original_test_index = sorted(test_forget_index + test_remain_index)

    original_train_set =  ConcatDataset([Subset(trainset_test, original_train_index), Subset(trainset_two_transform, original_train_index)])
    original_test_set = Subset(testset, original_test_index) 
    N = len(original_train_set)
    breakpoint()
    # breakpoint()
    binary_train_loader = DataLoader(dataset=original_train_set, batch_size=args.batch_size, shuffle=True)
    binary_test_loader =  DataLoader(dataset=original_test_set, batch_size=args.batch_size, shuffle=False)
    print(f"*** Dataloader successfully implemented: train:{len(original_train_set)} test:{len(original_test_set)} ***")


    '''
    Preparing Model. Backbone and probe
    '''
    model = torch.load(args.model_path, map_location='cpu').to(args.device) #should get_embeddings == True [1]
    for param in model.parameters():
        param.requires_grad = False
    
    # sample = model(torch.randn(52, 3, 32, 32).cuda(), get_embeddings=True)[1]
    # sample = sample.cuda()

    linear_probe = initialize_probe(args)
    print(f"*** Model, Probe successfully implemented ***")

    loss_fn = nn.CrossEntropyLoss()  # binary cross entropy
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=0.001)
    # max_iter = 50000 #6250
    max_iter = 6000
    data_gen = inf_generator(binary_train_loader)

    final_report = None
    #Train binary classification task with given loss
    for itr in tqdm(range(1, max_iter+1), desc='[Batch Training]'):
        x_train, y_train = data_gen.__next__()
        x_train, y_train = x_train.to(args.device), y_train.to(args.device)
        # y_train = F.one_hot(y_train, num_classes=2).float()

        with torch.no_grad():
            _ , embeddings = model(x_train, get_embeddings=True)
        logits = linear_probe(embeddings)

        ce_loss = loss_fn(logits, y_train)
        total_loss = ce_loss + (linear_probe.kl_divergence() / N)

        # linear_probe.zero_grad()
        optimizer.zero_grad()
        total_loss.backward() 
        optimizer.step()

        for layer in linear_probe.kl_list:
                layer.clip_variances()

        if itr % 1000 == 0:
            print('*** Eval period, doing eval ***')
            eval(itr, model, linear_probe, binary_test_loader, args)
            # linear_probe.zero_grad()
            linear_probe.train()

            print(f'{float(linear_probe.kl_divergence().detach().cpu().numpy()):.2f} : KL divergence')

        if itr == max_iter:
            print("*** train finished, storing result ***")
            kl_div = 0
            train_xent = 0
            linear_probe.eval()
            for idx, (inputs, labels) in enumerate(tqdm(binary_train_loader)):
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                with torch.no_grad():
                    _ , embeddings = model(inputs, get_embeddings=True)
                logits = linear_probe(embeddings)
                ce_loss = loss_fn(logits, labels)
                train_xent += ce_loss.detach().cpu().numpy() * inputs.shape[0]
            
            kl_div = float(linear_probe.kl_divergence().detach().cpu().numpy())
            final_acc = eval(itr, model, linear_probe, binary_test_loader, args)
            final_report = {
                    'train_xent': train_xent,
                    'kl': kl_div, 
                    'final_acc': final_acc  
            }
             
            with open(f'./train_report_{args.note}.json', 'w') as f:
                json.dump(final_report, f)

            #save model
            torch.save(linear_probe, f'./metric/probe_{args.note}.pth')
            

    #Evaluate Result
    # uniform_codelength = N * np.log2(2)
    uniform_codelength = N 
    model_bits, data_bits = final_report['kl'], final_report['train_xent']
    final_accuracy = final_report['final_acc']
    total_code = model_bits + data_bits

    print("***********************************************************")
    print(f"Final Acc: {final_accuracy:.3f}%")
    print(f"Model code: {round(model_bits)}bits | Data code: {round(data_bits)}bits")
    print("Variational codelength: {} bits".format(round(total_code)))
    print("Compression: {} ".format(round(uniform_codelength / data_bits, 2)))
    print("***********************************************************")
    

if __name__ == '__main__':
    args = arg_parse()
    args.model_path = './checkpoints/ResNet18_cifar10_ori.pth'
    # args.model_path = 'checkpoints/ResNet18_cifar10_class_4_5000_retrain.pth' 
    # args.model_path = './checkpoints/ResNet18_cifar100_class_4_500_retrain.pth'
    seed_torch(args.rnd_seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
