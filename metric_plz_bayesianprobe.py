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
        self.label_space_size = 10
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

def eval_original(current_itr, model, probe, test_loader, eval_type, args):
    forget_correct = 0
    forget_num = 0
    remain_correct = 0
    remain_num = 0

    correct = 0
    probe.eval()
    for idx, (imputs, labels) in enumerate(tqdm(test_loader)):
        inputs, labels = imputs.to(args.device), labels.to(args.device)
        with torch.no_grad():
            _ , embeddings = model(inputs, get_embeddings=True)
        output = probe(embeddings)
        pred = output.data.max(1)[1]
        current_correct = pred.eq(labels.data.view_as(pred)).cpu().sum().item()
        correct += current_correct

        label_forget_num = len(torch.where(labels == args.class_idx)[0])
        forget_num += label_forget_num
        remain_num += len(labels) - label_forget_num

        got_forget = len(torch.where(pred.eq(labels.data.view_as(pred)) * labels == args.class_idx)[0])
        forget_correct += got_forget
        remain_correct += current_correct - got_forget
    
    
    print(f"{eval_type} Acc: {((correct/len(test_loader.dataset))*100):.2f}% | ACC remain: {((remain_correct/remain_num)*100):.2f}% | ACC forget: {((forget_correct/forget_num)*100):.2f}% | itr: {current_itr}")
    
    total_acc = (correct/len(test_loader.dataset))*100 
    remain_acc = (remain_correct/remain_num)*100
    forget_acc = (forget_correct/forget_num) * 100
    
    return total_acc, remain_acc, forget_acc

def calculate_bayesian_data_code(current_itr, model, probe, train_loader, args):
    loss_fn = nn.CrossEntropyLoss()
    bin_num = 0
    if args.data_name == 'cifar10':
        bin_num = 10
    elif args.data_name == 'cifar100':
        bin_num = 100
    else: 
        raise NotImplementedError
    
    code_container = torch.zeros(bin_num)
    probe.eval()
    unlearn_num = 0
    for idx, (inputs, labels) in enumerate(tqdm(train_loader)):
        idx_list = []
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        with torch.no_grad():
                _ , embeddings = model(inputs, get_embeddings=True)
        logits = probe(embeddings)

        for idx in range(bin_num):
            cls_idx = torch.where(labels == idx)[0]
            idx_list.append(cls_idx)
            if idx == args.class_idx:
                unlearn_num += len(cls_idx)
        
        for id, tar_idx in enumerate(idx_list):
            tar_num = len(tar_idx)
            if tar_num != 0:
                tar_loss = loss_fn(logits[tar_idx], labels[tar_idx])
                tar_code = tar_loss.detach().item() * tar_num
                code_container[id] = code_container[id] + tar_code
            
    remain_num = len(train_loader.dataset) - unlearn_num
    forget_code = code_container[args.class_idx] 
    remain_code_sum = torch.sum(code_container) - forget_code

    avg_remain_code = (remain_code_sum / 9).item()
    avg_forget_code = (forget_code).item()
    kl_div = float(probe.kl_divergence().detach().cpu().numpy())
    total_code = torch.sum(code_container).item()

    code_list = code_container.tolist()
    print(f'***** Show Code length iter: {current_itr} *****')
    for idx, code in enumerate(code_list):
        print(f"Code length for class {idx} :  {round(code)} bits") 
    
    print(f"Avg remain code: {round(avg_remain_code)} | Avg forget code: {round(avg_forget_code)}")
    print(f"Model code: {round(kl_div)}")
    print(f'*****************************')
    
    return total_code, avg_remain_code, avg_forget_code, kl_div

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
    total_train_index = sorted(forget_index + remain_index)

    test_forget_index, test_remain_index = split_class_data(testset, args.class_idx, math.inf)
    total_test_index = sorted(test_forget_index + test_remain_index)

    total_train_set =  Subset(trainset_test, total_train_index)
    total_test_set = Subset(testset, total_test_index) 
    N = len(total_train_set)

    total_train_loader = DataLoader(dataset=total_train_set, batch_size=args.batch_size, shuffle=True)
    total_test_loader =  DataLoader(dataset=total_test_set, batch_size=args.batch_size, shuffle=False)
    print(f"*** Dataloader successfully implemented: train:{len(total_train_set)} test:{len(total_test_set)} ***")
    print(f"*** Dataset:{args.data_name} ***")


    '''
    Preparing Model. Backbone and probe
    '''
    model = torch.load(args.model_path, map_location='cpu').to(args.device) #should get_embeddings == True [1]
    for param in model.parameters():
        param.requires_grad = False

    linear_probe = initialize_probe(args)
    print(f"*** Model, Probe successfully implemented ***")

    loss_fn = nn.CrossEntropyLoss()  #cross entropy
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=0.001)
    
    max_iter = 0
    if args.data_name == 'cifar10':
        max_iter = 30000
    else:
        max_iter = 10000
    data_gen = inf_generator(total_train_loader)

    final_report = None
    #Train binary classification task with given loss
    for itr in tqdm(range(1, max_iter+1), desc='[Batch Training]'):
        x_train, y_train = data_gen.__next__()
        x_train, y_train = x_train.to(args.device), y_train.to(args.device)
        y_train = F.one_hot(y_train, num_classes=10).float()

        with torch.no_grad():
            _ , embeddings = model(x_train, get_embeddings=True)
        logits = linear_probe(embeddings)
        ce_loss = loss_fn(logits, y_train)
        total_loss = ce_loss + (linear_probe.kl_divergence() / N) 

        # linear_probe.zero_grad()
        optimizer.zero_grad()
        total_loss.backward() 
        optimizer.step()


        if itr % 1000 == 0:
            print('*** Eval period, doing eval ***')
            eval_original(itr, model, linear_probe, total_train_loader, 'Train', args)
            calculate_bayesian_data_code(itr, model, linear_probe, total_train_loader, args)
            linear_probe.train()

        if itr == max_iter:
            print("*** train finished, storing result ***")
            total_acc, remain_acc, forget_acc = eval_original(itr, model, linear_probe, total_train_loader, 'Train', args)
            total_code, avg_remain_code, avg_forget_code, kl_div = calculate_bayesian_data_code(itr, model, linear_probe, total_train_loader, args)
            final_report = {
                 'total_acc': total_acc,
                 'remain_acc': remain_acc,
                 'forget_acc': forget_acc,
                 'total_code': total_code,
                 'avg_remain_code': avg_remain_code, 
                 'avg_forget_code': avg_forget_code,
                 'kl_div': kl_div
,            }
             
            with open(f'./auxillary_train_report_{args.note}.json', 'w') as f:
                json.dump(final_report, f)

            #save model
            torch.save(linear_probe, f'./metric/auxillary_probe_{args.note}.pth')
            

    #Evaluate Result
    # uniform_codelength = N * np.log2(2)
    # final_accuracy, final_train_accuracy = final_report['final_test_acc'], final_report['final_train_acc']

    # print("***********************************************************")
    # print(f"Final Test Acc: {final_accuracy:.3f}% | Final Train Acc: {final_train_accuracy:.3f}")
    # print(f"Model code: {round(model_bits)}bits | Data code: {round(data_bits)}bits")
    # print("Variational codelength: {} bits".format(round(total_code)))
    # print("Compression: {} ".format(round(uniform_codelength / data_bits, 2)))
    # print("***********************************************************")
    

if __name__ == '__main__':
    args = arg_parse()
    # args.model_path = './checkpoints/ResNet18_cifar10_ori.pth'
    args.model_path = 'checkpoints/ResNet18_cifar10_class_4_5000_retrain.pth' 
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

    # args.model_path = './checkpoints/distill_cifar10_class_4_5000_1.0_2024_first_trial_1500.pth'
    # args.model_path = './checkcheck/ResNet18_cifar10_adam_seed6_retrain.pth'
    # args.model_path = './checkpoints/distill_cifar100_class_4_500_1.0_2024_2000.pth'
    seed_torch(0)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
