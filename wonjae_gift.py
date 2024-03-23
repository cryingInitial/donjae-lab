import os
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from backbone import ResNet
import numpy as np
from torch.utils.data import ConcatDataset, Subset, Dataset
from tqdm import tqdm


class ResNet_wrapper(nn.Module):
    def __init__(self, model, args):
        super(ResNet_wrapper, self).__init__()
        self.model = model
        self.to(args.device)

    def forward(self, x, get_all_features=False):
        
        all_embeddings = {}
        out = F.relu(self.model.bn1(self.model.conv1(x)))
        out = self.model.layer1(out)
        all_embeddings['l1'] = out
        out = self.model.layer2(out)
        all_embeddings['l2'] = out
        out = self.model.layer3(out)
        all_embeddings['l3'] = out
        out = self.model.layer4(out)
        all_embeddings['l4'] = out
        out = F.avg_pool2d(out, 4)
        embeddings = out.view(out.size(0), -1)
        all_embeddings['final'] = embeddings
        out = self.model.linear(embeddings)

        if get_all_features: return out, all_embeddings
        return out

def get_rank_info(model, probe, data_loader, args):

    probe.eval()
    fg_idx = rm_idx = 0
    fg_features_container, rm_features_container = [], [] 
    forget_indices = list(range(args.class_idx, args.class_idx + args.class_idx_unlearn))

    for idx, (inputs, labels) in enumerate(tqdm(data_loader)):
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        mask = torch.zeros_like(labels, dtype=torch.bool).to(args.device)
        for cls in forget_indices:
            mask = mask | (labels == cls)
        
        labels[mask] = 0
        labels[~mask] = 1

        one_hot_labels = F.one_hot(labels, num_classes=2).float() 
        fg_indices, rm_indices = torch.where(one_hot_labels[:,0]==1)[0], torch.where(one_hot_labels[:,1]==1)[0]

        with torch.no_grad():
            _ , all_embeddings = model(inputs, get_all_features=True)

        if len(fg_indices) > 0:
            fg_features =  all_embeddings['final'][fg_indices]
            fg_features_container[idx][fg_idx : fg_idx+len(fg_indices)] = fg_features
            fg_idx += len(fg_indices)
        if len(rm_indices) > 0:
            rm_features = all_embeddings['final'][rm_indices]
            rm_features_container[idx][rm_idx : rm_idx+len(rm_indices)] = rm_features
            rm_idx += len(rm_indices)


    forget_matrics, remain_matrices = fg_features_container[0], rm_features_container[0]

    fg_only_rank = calculate_feature_rank(forget_matrics) 
    rm_only_rank = calculate_feature_rank(remain_matrices)

    together_matrices = torch.cat((forget_matrics, remain_matrices))
    together_rank = calculate_feature_rank(together_matrices) #합집합 
    rank_overlap = fg_only_rank.item() + rm_only_rank.item() - together_rank.item() 

    result_dict = {
        'fg_rank': fg_only_rank.item(),
        'rm_rank': rm_only_rank.item(),
        'together_rank': together_rank.item(),
        'overlap_rank': rank_overlap 
    }
    
    return result_dict



def calculate_feature_rank(features):
    '''
    This function is implemented from RankMe
    we have NxK shaped features (which are embeddings. Thus no 3rd dimension)
    Here, Min(N,K) will be the maximum rank
    '''
    eps = 1e-10
    sig_values = torch.linalg.svd(features, full_matrices=False)[1]
    sig_values = sig_values / sig_values.sum() + eps
    features_rank = torch.exp((sig_values * -torch.log(sig_values)).sum())

    return features_rank