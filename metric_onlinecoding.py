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

from data import get_dataset, get_dataloader, get_unlearn_loader, get_forget_remain_loader#, split_class_data
from data import split_class_data as split_class_data_
from backbone import get_model
from method import run_method
from trainer import train_and_save
from eval import evaluate_summary
from util import report_sample_by_class
import BayesianLayers
from torch.utils.data import ConcatDataset, Subset
import copy

import time

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

def split_class_data(dataset, num_classes=10):
    index = {i:[] for i in range(num_classes)}
    for i, (data, target) in enumerate(dataset):
        index[target].append(i)
    return index

def split_data_into_portions(data_indices, forget_class, fractions):
    train_portions = []
    eval_portions = []
    forget_data_len = len(data_indices[forget_class])
    for i in range(len(fractions)):
        if i == 0:
            first_target_num = int(fractions[i]*forget_data_len)
        indices = []
        for values in data_indices.values():
            indices.extend(values[:int(fractions[i]*forget_data_len)])
        train_portions.append(indices[:])

        
        if i != len(fractions) - 1:
            indices= [] 
            for values in data_indices.values():
                indices.extend(values[int(fractions[i]*forget_data_len):int(fractions[i+1]*forget_data_len)])
            eval_portions.append(indices[:])
        else:
            eval_portions.append(indices[:])

    return train_portions, eval_portions, first_target_num

def main(args):
    #Prepare Dataloader
    '''
    Preparing balanced train, test loader
    '''
    max_epochs=20

    trainset, testset, trainset_test, trainset_two_transform, num_classes = get_dataset(args)

    # randomly shuffle 
    seed = args.seed
    trainset = Subset(trainset, np.random.default_rng(seed=seed).permutation(len(trainset)))
    trainset_test = Subset(trainset_test, np.random.default_rng(seed=seed).permutation(len(trainset_test)))

    train_class_indices = split_class_data(trainset, args.num_classes)

    fractions = [0.001,0.002,0.004,0.008,0.016,0.032,0.0625,0.125,0.25,0.5,1]
    train_indices, eval_indices, first_target_num = split_data_into_portions(train_class_indices, args.class_idx, fractions)
    # test_class_indices = split_class_data(testset, args.num_classes)

    # model = torch.load(args.model_path, map_location='cpu').to(args.device) #should get_embeddings == True [1]

    # random model
    model = get_model().to(args.device)

    for param in model.parameters():
        param.requires_grad_(False)

    testloader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)

    train_loss_per_cls_per_fraction = []
    eval_loss_per_cls_per_fraction = []
    test_loss_per_cls_per_fraction = []
    
    for i in range(len(train_indices)):
        # current_train_dataset = Subset(trainset, train_indices[i])
        current_train_dataset = Subset(trainset_test, train_indices[i])

        current_dataloader = DataLoader(dataset=current_train_dataset, batch_size=args.batch_size, shuffle=True)

        current_traintest_dataloader = DataLoader(dataset=Subset(trainset_test, train_indices[i]), batch_size=args.batch_size, shuffle=False)
        current_eval_dataloader = DataLoader(dataset=Subset(trainset_test, eval_indices[i]), batch_size=args.batch_size, shuffle=False)

        # reset head
        model.linear = nn.Linear(model.linear.in_features, model.linear.out_features).to(args.device)
        for param in model.linear.parameters():
            param.requires_grad_(True)

        best_model = model.state_dict()
        best_loss = 99999

        # optimizer = torch.optim.SGD(model.fc.parameters(), lr=)
        optimizer = torch.optim.Adam(model.linear.parameters(), lr=1e-3)

        # train the subset
        best_cnt = 0
        for epoch in range(max_epochs):
            model.train()
            for idx, (x, y) in enumerate((current_dataloader)):
                x, y = x.to(args.device), y.to(args.device)
                logit = model(x)
                ce_loss = F.cross_entropy(logit, y)
                optimizer.zero_grad()
                ce_loss.backward()
                optimizer.step()
        
            #test per epoch
            model.eval()
            val_loss = 0
            sample_num = 0
            for idx, (x, y) in enumerate((testloader)):
                x, y = x.to(args.device), y.to(args.device)
                logit = model(x)
                loss = F.cross_entropy(logit, y)
                val_loss += loss
                sample_num += y.shape[0]

            if val_loss / sample_num < best_loss:
                best_loss = val_loss / sample_num
                print(best_loss)
                best_model = copy.deepcopy(model.state_dict())
                best_cnt = 0
            else:
                best_cnt += 1
            if best_cnt >= 4:
                print("early stop")
                break
        model.load_state_dict(best_model)
        # get codelength

        # codelength & acc on train data
        correct = [0,]*args.num_classes
        total = [0,]*args.num_classes
        losses = [0,]*args.num_classes
        for idx, (x, y) in enumerate((current_traintest_dataloader)):
            x, y = x.to(args.device), y.to(args.device)
            logit = model(x)
            loss = F.cross_entropy(logit, y, reduction='none') / torch.log(torch.tensor([2])).to(args.device)
            for cls in range(args.num_classes):
                idx = torch.where(y==cls)[0]
                correct[cls] += (torch.argmax(logit[idx], dim=1) == y[idx]).sum()
                total[cls] += len(idx)
                losses[cls] += torch.sum(loss[idx].detach().cpu())
        
        print(f"total train acc: {(sum(correct)/sum(total)*100):.3f}% | forget train acc: {(correct[args.class_idx]/total[args.class_idx]*100):.3f}% | remain train acc: {(sum(correct) - correct[args.class_idx]) / (sum(total) - total[args.class_idx])*100:.3f}%")
        print(f"avg train loss: {(sum(losses)/sum(total)):.3f} | forget train avg loss: {(losses[args.class_idx]/total[args.class_idx]):.3f} | remain train avg loss: {(sum(losses) - losses[args.class_idx]) / (sum(total) - total[args.class_idx]):.3f}")

        train_loss_per_cls_per_fraction.append(losses[:])

        # codelength & acc on train data
        correct = [0,]*args.num_classes
        total = [0,]*args.num_classes
        losses = [0,]*args.num_classes
        for idx, (x, y) in enumerate((current_eval_dataloader)):
            x, y = x.to(args.device), y.to(args.device)
            logit = model(x)
            loss = F.cross_entropy(logit, y, reduction='none') / torch.log(torch.tensor([2])).to(args.device)
            for cls in range(args.num_classes):
                idx = torch.where(y==cls)[0]
                correct[cls] += (torch.argmax(logit[idx], dim=1) == y[idx]).sum()
                total[cls] += len(idx)
                losses[cls] += torch.sum(loss[idx].detach().cpu())
        
        print(f"total eval acc: {(sum(correct)/sum(total)*100):.3f}% | forget eval acc: {(correct[args.class_idx]/total[args.class_idx]*100):.3f}% | remain eval acc: {(sum(correct) - correct[args.class_idx]) / (sum(total) - total[args.class_idx])*100:.3f}%")
        print(f"total eval loss: {(sum(losses)):.3f} | forget eval total loss: {(losses[args.class_idx]):.3f} | remain eval total loss: {(sum(losses) - losses[args.class_idx]):.3f}")

        eval_loss_per_cls_per_fraction.append(losses[:])

        # codelength & acc on validation data
        correct = [0,]*args.num_classes
        total = [0,]*args.num_classes
        losses = [0,]*args.num_classes
        for idx, (x, y) in enumerate((testloader)):
            x, y = x.to(args.device), y.to(args.device)
            logit = model(x)
            loss = F.cross_entropy(logit, y, reduction='none') / torch.log(torch.tensor([2])).to(args.device)
            for cls in range(args.num_classes):
                idx = torch.where(y==cls)[0]
                correct[cls] += (torch.argmax(logit[idx], dim=1) == y[idx]).sum()
                total[cls] += len(idx)
                losses[cls] += torch.sum(loss[idx].detach().cpu())
        
        print(f"total test acc: {(sum(correct)/sum(total)*100):.3f}% | forget test acc: {(correct[args.class_idx]/total[args.class_idx]*100):.3f}% | remain test acc: {(sum(correct) - correct[args.class_idx]) / (sum(total) - total[args.class_idx])*100:.3f}%")
        print(f"avg test loss: {(sum(losses)/sum(total)):.3f} | forget test avg loss: {(losses[args.class_idx]/total[args.class_idx]):.3f} | remain test avg loss: {(sum(losses) - losses[args.class_idx]) / (sum(total) - total[args.class_idx]):.3f}")
        test_loss_per_cls_per_fraction.append(losses[:])
    # print per class codelength
    print("########original definition#########")
    for cls in range(args.num_classes):
        code_length = np.log2(args.num_classes) * first_target_num + sum(eval_loss[cls] for eval_loss in eval_loss_per_cls_per_fraction[:-1])
        total_trained_code_length = eval_loss_per_cls_per_fraction[-1][cls]
        model_length = code_length - total_trained_code_length
        print(f"cls {cls} | data code length : {code_length} | model code length : {model_length} ({total_trained_code_length})")
    
    torch.save({
        "train_loss_per_cls_per_fraction": train_loss_per_cls_per_fraction,
        "eval_loss_per_cls_per_fraction": eval_loss_per_cls_per_fraction,
        "test_loss_per_cls_per_fraction": test_loss_per_cls_per_fraction,
    # }, f"./losses_noaug/losses_seed{seed}_{(args.model_path).split('/')[-1]}")
    }, f"./losses_noaug/losses_seed{seed}_{(args.model_path).split('/')[-1]}")
    
    # print("#######loss on validation set########")
    # for cls in range(args.num_classes):
    #     code_length = sum(test_loss[cls] for test_loss in test_loss_per_cls_per_fraction)
    # breakpoint()

def main_binary(args):
    #Prepare Dataloader
    '''
    Preparing balanced train, test loader
    '''
    max_epochs=20

    trainset, testset, trainset_test, trainset_two_transform, num_classes = get_dataset(args)

    # forget_index, remain_index = split_class_data_(trainset, args.class_idx, args.class_unlearn)
    index = split_class_data(trainset, args.num_classes)
    forget_index= index[args.class_idx]
    remain_index = index[7]
    remain_index = np.random.choice(np.array(remain_index), len(forget_index)).tolist()
    binary_train_index = sorted(forget_index + remain_index)

    # test_forget_index, test_remain_index = split_class_data_(testset, args.class_idx, math.inf)
    test_index = split_class_data(testset, args.num_classes)
    test_forget_index= test_index[args.class_idx]
    test_remain_index= test_index[7]
    test_remain_index = np.random.choice(np.array(test_remain_index), len(test_forget_index)).tolist()
    binary_test_index = sorted(test_forget_index + test_remain_index)

    binary_train_set = Subset(trainset_test, binary_train_index)
    # binary_train_set = Subset(trainset, binary_train_index)

    binary_traintest_set = Subset(trainset_test, binary_train_index)
    binary_test_set = Subset(testset, binary_test_index)
    N = len(binary_train_set)

    binary_train_loader = DataLoader(dataset=binary_train_set, batch_size=args.batch_size, shuffle=True)
    binary_test_loader =  DataLoader(dataset=binary_test_set, batch_size=args.batch_size, shuffle=False)
    binary_train_testloader =  DataLoader(dataset=binary_traintest_set, batch_size=args.batch_size, shuffle=False)


    model = torch.load(args.model_path, map_location='cpu').to(args.device) #should get_embeddings == True [1]
    for param in model.parameters():
        param.requires_grad_(False)

    # reset head
    model.linear = nn.Linear(model.linear.in_features, 1).to(args.device)
    for param in model.linear.parameters():
        param.requires_grad_(True)

    best_model = model.state_dict()
    best_loss = 99999

    # optimizer = torch.optim.SGD(model.fc.parameters(), lr=)
    optimizer = torch.optim.Adam(model.linear.parameters(), lr=1e-3)

    # train the subset
    best_cnt = 0
    for epoch in range(max_epochs):
        model.train()
        for idx, (x, y) in enumerate((binary_train_loader)):
            x, y = x.to(args.device), torch.where(y == args.class_idx, 0, 1).to(args.device)
            logit = model(x)
            ce_loss = F.binary_cross_entropy_with_logits(logit, y.unsqueeze(-1).float())
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()
    
        #test per epoch
        model.eval()
        val_loss = 0
        sample_num = 0
        for idx, (x, y) in enumerate((binary_test_loader)):
            x, y = x.to(args.device), torch.where(y == args.class_idx, 0, 1).to(args.device)
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y.unsqueeze(-1).float())
            val_loss += loss
            sample_num += y.shape[0]

        if val_loss / sample_num < best_loss:
            best_loss = val_loss / sample_num
            print(best_loss)
            best_model = copy.deepcopy(model.state_dict())
            best_cnt = 0
        else:
            best_cnt += 1
        if best_cnt >= 4:
            print("early stop")
            break
    model.load_state_dict(best_model)
    # get codelength

    # codelength & acc on train data
    correct = [0,]*2
    total = [0,]*2
    train_losses = [0,]*2
    for idx, (x, y) in enumerate((binary_train_testloader)):
        x, y = x.to(args.device), torch.where(y == args.class_idx, 0, 1).to(args.device)
        logit = model(x)
        loss = F.binary_cross_entropy_with_logits(logit, y.unsqueeze(-1).float(), reduction='none') / torch.log(torch.tensor([2])).to(args.device)

        idx = torch.where(y==0)[0]
        correct[0] += (logit[idx] < 0.5).sum()
        total[0] += len(idx)
        train_losses[0] += torch.sum(loss[idx].detach().cpu())

        idx = torch.where(y==1)[0]
        correct[1] += (logit[idx] >= 0.5).sum()
        total[1] += len(idx)
        train_losses[1] += torch.sum(loss[idx].detach().cpu())
    
    print(f"total train acc: {(sum(correct)/sum(total)*100):.3f}% | forget train acc: {(correct[0]/total[0]*100):.3f}% | remain train acc: {(sum(correct) - correct[0]) / (sum(total) - total[0])*100:.3f}%")
    print(f"total train loss: {(sum(train_losses)):.3f} | forget train total loss: {(train_losses[0]):.3f} | remain train total loss: {(sum(train_losses) - train_losses[0]):.3f} | ratio: {train_losses[0]/train_losses[1]}")

    # codelength & acc on train data
    correct = [0,]*2
    total = [0,]*2
    test_losses = [0,]*2
    for idx, (x, y) in enumerate((binary_test_loader)):
        x, y = x.to(args.device), torch.where(y == args.class_idx, 0, 1).to(args.device)
        logit = model(x)
        loss = F.binary_cross_entropy_with_logits(logit, y.unsqueeze(-1).float(), reduction='none') / torch.log(torch.tensor([2])).to(args.device)
        idx = torch.where(y==0)[0]
        correct[0] += (logit[idx] < 0.5).sum()
        total[0] += len(idx)
        test_losses[0] += torch.sum(loss[idx].detach().cpu())

        idx = torch.where(y==1)[0]
        correct[1] += (logit[idx] >= 0.5).sum()
        total[1] += len(idx)
        test_losses[1] += torch.sum(loss[idx].detach().cpu())
    
    print(f"total test acc: {(sum(correct)/sum(total)*100):.3f}% | forget test acc: {(correct[0]/total[0]*100):.3f}% | remain test acc: {(sum(correct) - correct[0]) / (sum(total) - total[0])*100:.3f}%")
    print(f"total test loss: {(sum(test_losses)):.3f} | forget test total loss: {(test_losses[0]):.3f} | remain test total loss: {(sum(test_losses) - test_losses[0]):.3f} | ratio: {test_losses[0]/test_losses[1]}")

def main_binary2(args):
    #Prepare Dataloader
    '''
    Preparing balanced train, test loader
    '''
    max_epochs=20

    trainset, testset, trainset_test, trainset_two_transform, num_classes = get_dataset(args)

    forget_index, remain_index = split_class_data_(trainset, args.class_idx, args.class_unlearn)
    remain_index = np.random.choice(np.array(remain_index), len(forget_index)).tolist()

    fractions = [0.001,0.002,0.004,0.008,0.016,0.032,0.0625,0.125,0.25,0.5,1]
    train_portions = []
    eval_portions = []
    first_target_num = len(forget_index)
    forget_data_len = len(forget_index)
    for i in range(len(fractions)):
        if i == 0:
            first_target_num = int(fractions[i]*forget_data_len)
        indices = []
        indices.extend(forget_index[:int(fractions[i]*forget_data_len)])
        indices.extend(remain_index[:int(fractions[i]*forget_data_len)])
        train_portions.append(indices[:])
        
        if i != len(fractions) - 1:
            indices= [] 
            indices.extend(forget_index[int(fractions[i]*forget_data_len):int(fractions[i+1]*forget_data_len)])
            indices.extend(remain_index[int(fractions[i]*forget_data_len):int(fractions[i+1]*forget_data_len)])
            eval_portions.append(indices[:])
        else:
            eval_portions.append(indices[:])

    test_forget_index, test_remain_index = split_class_data_(testset, args.class_idx, math.inf)
    test_remain_index = np.random.choice(np.array(test_remain_index), len(test_forget_index)).tolist()
    binary_test_index = sorted(test_forget_index + test_remain_index)

    binary_test_set = Subset(testset, binary_test_index)

    binary_test_loader =  DataLoader(dataset=binary_test_set, batch_size=args.batch_size, shuffle=False)

    # model = torch.load(args.model_path, map_location='cpu').to(args.device) #should get_embeddings == True [1]

    # random model
    model = get_model().to(args.device)

    for param in model.parameters():
        param.requires_grad_(False)

    losses = []
    for i in range(len(train_portions)):
        # current_train_dataset = Subset(trainset, train_portions[i])
        current_train_dataset = Subset(trainset_test, train_portions[i])

        current_dataloader = DataLoader(dataset=current_train_dataset, batch_size=args.batch_size, shuffle=True)

        current_eval_dataloader = DataLoader(dataset=Subset(trainset_test, eval_portions[i]), batch_size=args.batch_size, shuffle=False)

        # reset head
        model.linear = nn.Linear(model.linear.in_features, 1).to(args.device)
        for param in model.linear.parameters():
            param.requires_grad_(True)

        best_model = model.state_dict()
        best_loss = 99999

        # optimizer = torch.optim.SGD(model.fc.parameters(), lr=)
        optimizer = torch.optim.Adam(model.linear.parameters(), lr=1e-3)

        # train the subset
        best_cnt = 0
        for epoch in range(max_epochs):
            model.train()
            for idx, (x, y) in enumerate((current_dataloader)):
                x, y = x.to(args.device), torch.where(y == args.class_idx, 0, 1).to(args.device)
                logit = model(x)
                ce_loss = F.binary_cross_entropy_with_logits(logit, y.unsqueeze(-1).float())
                optimizer.zero_grad()
                ce_loss.backward()
                optimizer.step()
        
            #test per epoch
            model.eval()
            val_loss = 0
            sample_num = 0
            for idx, (x, y) in enumerate((binary_test_loader)):
                x, y = x.to(args.device), torch.where(y == args.class_idx, 0, 1).to(args.device)
                logit = model(x)
                loss = F.binary_cross_entropy_with_logits(logit, y.unsqueeze(-1).float())
                val_loss += loss
                sample_num += y.shape[0]

            if val_loss / sample_num < best_loss:
                best_loss = val_loss / sample_num
                print(best_loss)
                best_model = copy.deepcopy(model.state_dict())
                best_cnt = 0
            else:
                best_cnt += 1
            if best_cnt >= 4:
                print("early stop")
                break
        model.load_state_dict(best_model)
        # get codelength

        # codelength & acc on train data
        correct = [0,]*2
        total = [0,]*2
        train_losses = [0,]*2
        for idx, (x, y) in enumerate((current_eval_dataloader)):
            x, y = x.to(args.device), torch.where(y == args.class_idx, 0, 1).to(args.device)
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y.unsqueeze(-1).float(), reduction='none') / torch.log(torch.tensor([2])).to(args.device)

            idx = torch.where(y==0)[0]
            correct[0] += (logit[idx] < 0.5).sum()
            total[0] += len(idx)
            train_losses[0] += torch.sum(loss[idx].detach().cpu())

            idx = torch.where(y==1)[0]
            correct[1] += (logit[idx] >= 0.5).sum()
            total[1] += len(idx)
            train_losses[1] += torch.sum(loss[idx].detach().cpu())
        
        print(f"total train acc: {(sum(correct)/sum(total)*100):.3f}% | forget train acc: {(correct[0]/total[0]*100):.3f}% | remain train acc: {(sum(correct) - correct[0]) / (sum(total) - total[0])*100:.3f}%")
        print(f"total train loss: {(sum(train_losses)):.3f} | forget train total loss: {(train_losses[0]):.3f} | remain train total loss: {(sum(train_losses) - train_losses[0]):.3f} | ratio: {train_losses[0]/train_losses[1]}")

        losses.append(sum(train_losses))

        # codelength & acc on train data
        correct = [0,]*2
        total = [0,]*2
        test_losses = [0,]*2
        for idx, (x, y) in enumerate((binary_test_loader)):
            x, y = x.to(args.device), torch.where(y == args.class_idx, 0, 1).to(args.device)
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y.unsqueeze(-1).float(), reduction='none') / torch.log(torch.tensor([2])).to(args.device)
            idx = torch.where(y==0)[0]
            correct[0] += (logit[idx] < 0.5).sum()
            total[0] += len(idx)
            test_losses[0] += torch.sum(loss[idx].detach().cpu())

            idx = torch.where(y==1)[0]
            correct[1] += (logit[idx] >= 0.5).sum()
            total[1] += len(idx)
            test_losses[1] += torch.sum(loss[idx].detach().cpu())
        
        print(f"total test acc: {(sum(correct)/sum(total)*100):.3f}% | forget test acc: {(correct[0]/total[0]*100):.3f}% | remain test acc: {(sum(correct) - correct[0]) / (sum(total) - total[0])*100:.3f}%")
        print(f"total test loss: {(sum(test_losses)):.3f} | forget test total loss: {(test_losses[0]):.3f} | remain test total loss: {(sum(test_losses) - test_losses[0]):.3f} | ratio: {test_losses[0]/test_losses[1]}")

    print("#### online code length ###")
    code_length = np.log2(2) * first_target_num + sum(eval_loss for eval_loss in losses[:-1])
    print(f"{code_length:.3f}")

def print_metrics(args):
     # model performance
    _, testset, trainset_test, _, num_classes = get_dataset(args)
    # model = torch.load(args.model_path, map_location='cpu').to(args.device)
    model = get_model().to(args.device)
    model.eval()
    trainloader = DataLoader(dataset=trainset_test, batch_size=args.batch_size, shuffle=False)
    testloader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)

    correct = [0,]*args.num_classes
    total = [0,]*args.num_classes
    for idx, (x, y) in enumerate(tqdm(trainloader)):
        x, y = x.to(args.device), y.to(args.device)
        logit = model(x)
        for cls in range(args.num_classes):
            idx = torch.where(y==cls)[0]
            correct[cls] += (torch.argmax(logit[idx], dim=1) == y[idx]).sum()
            total[cls] += len(idx)
    print(f"total train acc: {(sum(correct)/sum(total)*100):.3f}% | forget train acc: {(correct[args.class_idx]/total[args.class_idx]*100):.3f}% | remain train acc: {(sum(correct) - correct[args.class_idx]) / (sum(total) - total[args.class_idx])*100:.3f}%")

    correct = [0,]*args.num_classes
    total = [0,]*args.num_classes
    for idx, (x, y) in enumerate(tqdm(testloader)):
        x, y = x.to(args.device), y.to(args.device)
        logit = model(x)
        for cls in range(args.num_classes):
            idx = torch.where(y==cls)[0]
            correct[cls] += (torch.argmax(logit[idx], dim=1) == y[idx]).sum()
            total[cls] += len(idx)
    print(f"total test acc: {(sum(correct)/sum(total)*100):.3f}% | forget test acc: {(correct[args.class_idx]/total[args.class_idx]*100):.3f}% | remain test acc: {(sum(correct) - correct[args.class_idx]) / (sum(total) - total[args.class_idx])*100:.3f}%")
    # code length measure
    
    losses = torch.load(f"./losses_noaug/losses_seed{seed}_{(args.model_path).split('/')[-1]}")
    first_target_num = 50

    eval_loss_per_cls_per_fraction = losses['eval_loss_per_cls_per_fraction']
    train_loss_per_cls_per_fraction = losses['train_loss_per_cls_per_fraction']
    test_loss_per_cls_per_fraction = losses['test_loss_per_cls_per_fraction']

    print(args.model_path)

    print("########original definition#########")
    random_code_length = 5000*np.log2(args.num_classes)
    print(random_code_length)
    code_lens = []
    for cls in range(args.num_classes):
        code_length = np.log2(args.num_classes) * first_target_num + sum(eval_loss[cls] for eval_loss in eval_loss_per_cls_per_fraction[:-1])
        code_lens.append(code_length)
        total_trained_code_length = eval_loss_per_cls_per_fraction[-1][cls]
        model_length = code_length - total_trained_code_length
        print(f"cls {cls} | data code length : {code_length} | avg code length: {code_length/5000} | model code length : {model_length} ({total_trained_code_length})")

    remain_code_length = (sum(code_lens) - code_lens[args.class_idx]) / (len(code_lens) - 1)
    print(f"remain code length: {remain_code_length} | forget code length: {code_lens[args.class_idx]} | ratio: {code_lens[args.class_idx] / remain_code_length}")
    
    remain_final_length = (sum(eval_loss_per_cls_per_fraction[-1][cls] for cls in range(args.num_classes)) - eval_loss_per_cls_per_fraction[-1][args.class_idx]) / 9
    print(f"remain final code length: {remain_final_length} | forget final code length: {eval_loss_per_cls_per_fraction[-1][args.class_idx]} | ratio: {eval_loss_per_cls_per_fraction[-1][args.class_idx] / remain_final_length}")
    
    # breakpoint()
    print("#####test data value#####")
    remain_test_code_length = 0
    for cls in range(args.num_classes):
        test_code_length = test_loss_per_cls_per_fraction[-1][cls]
        print(f"cls {cls} | test data code length: {test_code_length}")
        remain_test_code_length += test_code_length
    remain_test_code_length = (remain_test_code_length - test_loss_per_cls_per_fraction[-1][args.class_idx])/ (args.num_classes - 1)
    print(f"remain test code length: {remain_test_code_length} | forget test code length: {test_loss_per_cls_per_fraction[-1][args.class_idx]} | ratio: {test_loss_per_cls_per_fraction[-1][args.class_idx]/remain_test_code_length}")
        


if __name__ == '__main__':
    args = arg_parse()
    # args.model_path = './checkcheck/ResNet18_cifar10_adam_seed5_retrain.pth'
    # args.model_path = './checkpoints/ResNet18_cifar10_ori.pth'
    # args.model_path = './checkpoints/ft_cifar10_class_4_5000_1.0_0.pth'
    # args.model_path = './checkpoints/ft_cifar10_class_4_5000_1.0_0_20000.pth'
    # args.model_path = './checkpoints/ft_cifar10_class_4_5000_1.0_0_50000.pth'
    # args.model_path = './checkpoints/teacher_cifar10_class_4_5000_1.0_adam_1e-52.pth'
    # args.model_path = "./checkpoints/scrub_cifar10_class_4_5000_1.0_adam_5e-51.pth"
    # args.model_path = "./checkpoints/pgd_cifar10_class_4_5000_1.0_adam_5e-42.pth"
    # args.model_path = "./checkpoints/distill_cifar10_class_4_5000_1.0_kd_kd_200_2.pth"
    args.model_path = "./checkpoints/distill_cifar10_class_4_5000_1.0_kd_kd_2000_2.pth"
    # args.model_path = "./checkpoints/distill_cifar10_class_4_5000_1.0_0_first_trial.pth"
    # args.model_path = "./checkpoints/distill_cifar10_class_4_5000_1.0_adam_1e-52.pth"

    print(args.model_path)
    seed=0
    args.seed = seed
    seed_torch(seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()
    # main(args)
    # main_binary(args)
    # main_binary2(args)
    print_metrics(args)

    end = time.time()

    print(f"runtime: {(end-start):.3f} sec")
   