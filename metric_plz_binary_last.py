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
import pickle
import json
from tqdm import tqdm
from adv_generator import inf_generator

from backbone import get_model
from method import run_method
from trainer import train_and_save
from eval import evaluate_summary
from util import report_sample_by_class
from torch.utils.data import ConcatDataset, Subset, Dataset
from metric_util import get_bayesian_probe, get_metric_dataset, binary_metric_log, ResNet_wrapper, get_binary_loader, seed_torch, get_wrapped_model, get_metric_optimizer, get_metric_loss, binary_eval, get_specific_ranks, get_multi_ranks, metric_eval

def arg_parse():
    parser = argparse.ArgumentParser("Boundary Unlearning")
    parser.add_argument('--rnd_seed', type=int, default=0, help='random seed') # 0, 1, 2
    parser.add_argument('--method', type=str, default='ft', help='unlearning method')
    parser.add_argument('--data_name', type=str, default='cifar100', help='dataset, cifar10, cifar100, imagenet, cars or flowers')
    parser.add_argument('--model_name', type=str, default='ResNet18', help='model name')
    parser.add_argument('--batch_size', type=int, default=250, help='batch size')
    parser.add_argument('--test_mode', type=str, default='sample', choices=['sample', 'class'], help='unlearning mode')
    parser.add_argument('--debug_mode', action='store_true', help='debug mode, it takes more time but with more detailed information')
    parser.add_argument('--save_result_model', action='store_true', help="save result model")
    parser.add_argument('--note', type=str, default='', help='note')
    parser.add_argument('--test_interval', type=int, default=-1, help='test interval')
    parser.add_argument('--retain_ratio', type=float, default=1, help='retain ratio')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')

    # model selection
    parser.add_argument('--unlearn_aug', action='store_true', help="unlearn with data augmentation")

    # class unlearning, test_mode=class
    parser.add_argument('--class_idx', type=int, default=4, help='class index to unlearn')
    parser.add_argument('--class_idx_unlearn', type=int, default=1, help='class index to unlearn')
    parser.add_argument('--class_unlearn', type=int, default=100, help='number of unlearning samples')

    # sample unlearning, test_mode=sample
    parser.add_argument('--sample_unlearn_per_class', type=int, default=100, help='number of unlearning samples per class')

    #bayesian arguments
    parser.add_argument('--aug_type', type=str, default='soft', help='aug type used to amplify dataset')
    parser.add_argument('--dataset_length', type=int, default=500, help='number of samples in a single class you want')
    parser.add_argument('--metric_task', type=str, default='binary', help='metric task in [binary, original]')
    parser.add_argument('--extractor_path', type=str, default='', help='checkpoint path of extractor')
    parser.add_argument('--probe_type', type=str, default='simple_fc', help='probe types in [simple_fc, heavy_fc, simple_conv]')
    parser.add_argument('--metric_lr',type=float, default=0.001, help='metric learning rate')
    parser.add_argument('--metric_optimizer', type=str, default='adam', help="metric optimizer in [adam, sgd]")
    args = parser.parse_args()

    return args


def main(args):
    #Prepare Dataloader
    '''
    Preparing balanced train, test loader
    '''
    binary_train_loader = binary_test_loader = None
    N = 2 * args.dataset_length

    train_forget_set, train_remain_set, test_forget_set, test_remain_set = get_metric_dataset(args)
    # randomly split train_forget_set, train_remain_set
    train_remain_set = Subset(train_remain_set, np.random.choice(len(train_remain_set), 5000, replace=False))
    binary_train_loader, binary_test_loader = get_binary_loader(train_forget_set, train_remain_set, test_forget_set, test_remain_set, args)
    train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader = DataLoader(train_forget_set, batch_size=args.batch_size, shuffle=False), DataLoader(train_remain_set, batch_size=args.batch_size, shuffle=False), DataLoader(test_forget_set, batch_size=args.batch_size, shuffle=False), DataLoader(test_remain_set, batch_size=args.batch_size, shuffle=False)
    assert len(binary_train_loader.dataset) == 2 * args.dataset_length
    print(f"*** Dataloader successfully implemented ***")
    print(f"*** Dataset:{args.data_name} ***")


    '''
    Preparing Model. Extractor and probe
    '''
    if args.extractor_path == '':
        args.logger.info("No checkpoint path. Random Extractor has initialized")
        extractor = get_model().to(args.device)
        extractor = get_wrapped_model(extractor, args) 
    else:
        extractor = torch.load(args.extractor_path, map_location='cpu').to(args.device)
        extractor = get_wrapped_model(extractor, args)
    
    for param in extractor.parameters():
        param.requires_grad = False
    
    
    #TODO: get input dim automatically by forwarding samples. Also, should make diversified model types
    if args.metric_task == 'binary': out_dim = 2 
    else: raise NotImplementedError
    probe = get_bayesian_probe(512, out_dim, args)
    args.logger.info('*** Extractor, Probe has implemented successfully ***')


    loss_fn = get_metric_loss(args)  # binary cross entropy
    optimizer = get_metric_optimizer(probe, args)

    if args.data_name == 'cifar10':
        max_iter = 50000 
    elif args.data_name == 'cifar100':
        max_iter = 40000
    else:
        max_iter = 50000

    data_gen = inf_generator(binary_train_loader)

    final_report = None
    #Train binary classification task with given loss
    forget_indices = list(range(args.class_idx, args.class_idx + args.class_idx_unlearn))
    for itr in tqdm(range(1, max_iter+1), desc='[Batch Training]'):
        x_train, y_train = data_gen.__next__()
        x_train, y_train = x_train.to(args.device), y_train.to(args.device)

        mask = torch.zeros_like(y_train, dtype=torch.bool).to(args.device)
        for cls in forget_indices:
            mask = mask | (y_train == cls)

        y_train[mask] = 0
        y_train[~mask] = 1

        y_train = F.one_hot(y_train, num_classes=2).float()

        with torch.no_grad():
            _ , embeddings = extractor(x_train, get_all_features=True)
        feats = embeddings['final'] 
        logits = probe(feats)

        bce_loss = loss_fn(logits, y_train)
        total_loss = bce_loss + (probe.kl_divergence() / N)

        # linear_probe.zero_grad()
        optimizer.zero_grad()
        total_loss.backward() 
        optimizer.step()

        for layer in probe.kl_list:
                layer.clip_variances()


        if itr % 10 == 0 or itr == max_iter:
            args.logger.info("**** eval period *****")
    
            rank_check = ['final']
            eval_result = binary_eval(itr, extractor, probe, binary_train_loader, 'train', rank_check, args)
            remain_result = metric_eval(extractor, train_remain_loader, args)
            forget_result = metric_eval(extractor, train_forget_loader, args)


            remain_multi_ranks = get_multi_ranks(remain_result, args)
            remain_result[4] = forget_result[4]
            full_multi_ranks = get_multi_ranks(remain_result, args)
            # print(f"remain_multi_ranks: {remain_multi_ranks}", f"full_multi_ranks: {full_multi_ranks}")

            eval_result2 = metric_eval(extractor, binary_train_loader, args)


            assert len(eval_result['fg_container']) == 1 and len(eval_result['rm_container']) == 1, 'Only final embeddings are workable'
            fg_features, rm_features = eval_result['fg_container'][0], eval_result['rm_container'][0]
            rank_result = get_specific_ranks(fg_features, rm_features, args)
            args.logger.info(f"h_score: {rank_result['h_score']}, remain_h_score: {remain_multi_ranks['h_score']}, forget_h_score: {full_multi_ranks['h_score']}")
            eval_result.update(rank_result)

            total_result = {'itr': itr}
            for k, v in eval_result.items():
                if k not in ['fg_container', 'rm_container']:
                    total_result[k] = v 

            json_path = os.path.join(args.log_path, f'{args.rnd_seed}_data.json')

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    existing_data = json.load(f)
                
                existing_data.append(total_result)
                with open(json_path, 'w') as f:
                    json.dump(existing_data, f)
            else:
                with open(json_path, 'a') as f:
                    json.dump([total_result], f)


            if itr == max_iter:
                torch.save(probe, f'./metric/{args.method_model_name}.pth')
                
                #generate final results
                with open(json_path, 'r') as f:
                    total_data = json.load(f)

            # probe.zero_grad()
            probe.train()
            
        
    #Evaluate Result
    # uniform_codelength = N * np.log2(2)
    # uniform_codelength = N 
    model_bits, data_bits = total_result['model_length'], total_result['data_length']
    final_accuracy, fg_rank, rm_rank, together_rank = final_report['acc'], final_report['fg_rank'], final_report['rm_rank'], final_report['together_rank']


    total_code = model_bits + data_bits
    args.logger.info("***********************************************************")
    args.logger.info(f"Final Train Acc: {final_accuracy:.3f}%")
    args.logger.info(f"Model code: {round(model_bits)}bits | Data code: {round(data_bits)}bits | Total code: {round(model_bits + data_bits)} ")
    args.logger.info(f"fg_rank: {round(fg_rank)} | rm_rank: {round(rm_rank)} | together_rank: {round(together_rank)} | overlap_rank: {round(fg_rank + rm_rank - together_rank)} | ratio: {round((fg_rank + rm_rank - together_rank)/(together_rank),3)}")
    args.logger.info("***********************************************************")
    

if __name__ == '__main__':
    args = arg_parse()
    seed_torch(args.rnd_seed)
    # args.extractor_path = 'checkpoints/distill_cifar10_class_4_5000_1.0_2024_first_trial_1500.pth'
    # args.extractor_path = 'checkpoints/ft_cifar10_class_4_5000_1.0_0_20000.pth'
    # args.extractor_path = 'checkpoints/ft_cifar10_class_4_5000_1.0_2024_2000.pth'
    # args.extractor_path = 'checkpoints/distill_cifar10_class_4_5000_1.0_0_first_trial.pth'
    # args.extractor_path = 'checkpoints/scrub_cifar10_class_4_5000_1.0_2024_2000.pth'
    # args.extractor_path = 'checkcheck/ResNet18_cifar10_adam_seed4_retrain.pth'
    # args.extractor_path = 'checkcheck/ResNet18_cifar10_adam_seed5_retrain.pth'
    # args.extractor_path = 'checkcheck/ResNet18_cifar10_adam_seed6_retrain.pth'

    binary_metric_log(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
