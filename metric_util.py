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
import BayesianLayers
from backbone import ResNet
import numpy as np
from torch.utils.data import ConcatDataset, Subset, Dataset
from tqdm import tqdm
import logging.config
from torch import optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

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

    def __init__(self, input_dim, output_dim, model_type, args):
        print('Constructing BayesProbe')
        super(BayesProbe, self).__init__()
        self.args = args
        self.model_type = model_type
        self.input_dim = input_dim
        self.label_space_size = output_dim
        #------------------------------------------------
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.kl_list = []

        if self.model_type == 'simple_fc':
            self.initial_layer = BayesianLayers.LinearGroupNJ(self.input_dim, 128, clip_var=0.04, cuda=True)
            self.last_layer = BayesianLayers.LinearGroupNJ(128, self.label_space_size, cuda=True)

            self.kl_list = [self.initial_layer, self.last_layer]

        elif self.model_type == 'heavy_fc':
            self.initial_layer = BayesianLayers.LinearGroupNJ(self.input_dim, 256, clip_var=0.04, cuda=True)
            self.intermediate_layers = nn.ModuleList()
            self.intermediate_layers.append(BayesianLayers.LinearGroupNJ(256, 128, cuda=True))
            self.last_layer = BayesianLayers.LinearGroupNJ(128, self.label_space_size, cuda=True)

            self.kl_list = [self.initial_layer] + [l for l in self.intermediate_layers] + [self.last_layer]

        else:
            raise NotImplementedError

        assert len(self.kl_list) != 0
        #------------------------------------------------
        self.to(args.device)
        self.print_param_count()

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.relu(self.initial_layer(x))

        if self.model_type != 'simple_fc':
            for layer in self.intermediate_layers:
                x = self.relu(layer(x))

        x = self.last_layer(x)
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

def seed_torch(seed):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_wrapped_model(model, args):

    if args.model_name.startswith('ResNet'):
        model = ResNet_wrapper(model, args)
    else:
        raise NotImplementedError
    
    return model
    
def binary_metric_log(args):
    logging.config.fileConfig('logging.conf')
    args.logger = logging.getLogger()
    os.makedirs(f'./metric_logs/{args.data_name}', exist_ok=True)
    if args.test_mode == 'class': args.method_model_name = f'{args.metric_task}_{args.note}_{args.test_mode}_{args.class_idx}_{args.dataset_length}_{args.aug_type}'
    elif args.test_mode == 'sample': args.method_model_name = f'{args.metric_task}_{args.note}_{args.test_mode}_{args.class_idx}_{args.dataset_length}_{args.aug_type}'
    os.makedirs(f'./metric_logs/{args.data_name}/{args.method_model_name}', exist_ok=True)
    fileHandler = logging.FileHandler(f'./metric_logs/{args.data_name}/{args.method_model_name}/seed{args.rnd_seed}.log', mode='w')
    args.logger.addHandler(fileHandler)
    args.log_path = f'./metric_logs/{args.data_name}/{args.method_model_name}'
    

def get_bayesian_probe(input_dim, output_dim, args):
    model_type = args.probe_type
    assert model_type in ['simple_fc', 'heavy_fc', 'simple_conv']
    return BayesProbe(input_dim, output_dim, model_type, args)

def get_metric_loss(args):
    task_type = args.metric_task
    if task_type == 'binary':
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    return loss_fn
    
def get_metric_optimizer(probe, args, momentum = 0.):
    optimizer, lr = args.metric_optimizer, args.metric_lr
    param = probe.parameters()
    if optimizer == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(param, lr=lr, momentum=momentum)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)
    return optimizer

def get_metric_dataset(args):
    mean, std, image_size, num_classes = get_statistics(args.data_name)
    args.mean, args.std, args.image_size, args.num_classes = mean, std, image_size, num_classes
    #Horizontal flip is excluded
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # if no dataset execute shell file
    if not os.path.exists(f'./dataset/{args.data_name}'):
        subprocess.run([f'./dataset/{args.data_name}.sh'])
        subprocess.run(['mv', f'./{args.data_name}', './dataset'])
        subprocess.run(['rm', f'./{args.data_name}_png.tar'])
    
    train_forget_set = FilteredDataset(f'./dataset/{args.data_name}/train', list(range(args.class_idx, args.class_idx + args.class_idx_unlearn)), transform=train_transform, exclude=False)
    train_remain_set =  FilteredDataset(f'./dataset/{args.data_name}/train', list(range(args.class_idx, args.class_idx + args.class_idx_unlearn)), transform=train_transform, exclude=True)
    # train_forget_set = FilteredDataset(f'./dataset/{args.data_name}/train', [5], transform=train_transform, exclude=False)
    # train_remain_set =  FilteredDataset(f'./dataset/{args.data_name}/train', [4.5], transform=train_transform, exclude=True)

    test_forget_set = FilteredDataset(f'./dataset/{args.data_name}/test', list(range(args.class_idx, args.class_idx + args.class_idx_unlearn)), transform=test_transform, exclude=False)
    test_remain_set =  FilteredDataset(f'./dataset/{args.data_name}/test', list(range(args.class_idx, args.class_idx + args.class_idx_unlearn)), transform=test_transform, exclude=True)

    return train_forget_set, train_remain_set, test_forget_set, test_remain_set

    
def get_binary_loader(train_forget_set, train_remain_set, test_forget_set, test_remain_set, args):
    '''
    dataset_length is number of samples for single class.
    '''
    dataset_length = args.dataset_length
    source_num = len(train_forget_set)
    assert len(train_remain_set) >= source_num, "remain set should be bigger than forget set"

    if len(train_forget_set) < dataset_length:
        THRES = 5000
        preprocess_flag = True if source_num >= THRES else False
        #if source_num is too small, then we do not select train_remain_set first but select after transform.
        if preprocess_flag:
            remain_idx = np.random.choice(np.arange(len(train_remain_set)), source_num).tolist()
            train_remain_set = Subset(train_remain_set, remain_idx)
    
        deficient_num = dataset_length - source_num
        needed_aug_num = round(deficient_num / source_num)
        aug_list = generate_different_transforms(needed_aug_num, args.aug_type, args)

        train_forget_container, train_remain_container = [train_forget_set], [train_remain_set]

        for transform in aug_list:
            fg_set = FilteredDataset(f'./dataset/{args.data_name}/train', list(range(args.class_idx, args.class_idx + args.class_idx_unlearn)), transform=transform, exclude=False)
            rm_set = FilteredDataset(f'./dataset/{args.data_name}/train', list(range(args.class_idx, args.class_idx + args.class_idx_unlearn)), transform=transform, exclude=True)
            # fg_set = FilteredDataset(f'./dataset/{args.data_name}/train', [4,5], transform=transform, exclude=False)
            # rm_set = FilteredDataset(f'./dataset/{args.data_name}/train', [4], transform=transform, exclude=True)

            if preprocess_flag:
                rm_set = Subset(rm_set, remain_idx)

            train_forget_container.append(fg_set)
            train_remain_container.append(rm_set)
        
        train_forget_set, train_remain_set = ConcatDataset(train_forget_container), ConcatDataset(train_remain_container)
        train_fg_idx = np.random.choice(np.arange(len(train_forget_set)), dataset_length).tolist()
        train_rm_idx = np.random.choice(np.arange(len(train_remain_set)), dataset_length).tolist()

        train_forget_set, train_remain_set = Subset(train_forget_set, train_fg_idx), Subset(train_remain_set, train_rm_idx)

    else:
        train_fg_idx = np.random.choice(np.arange(len(train_forget_set)), dataset_length).tolist()
        train_rm_idx = np.random.choice(np.arange(len(train_remain_set)), dataset_length).tolist()
        train_forget_set, train_remain_set = Subset(train_forget_set, train_fg_idx), Subset(train_remain_set, train_rm_idx)


    #set test remain set as test forget set
    test_rm_idx = np.random.choice(np.arange(len(test_remain_set)), len(test_forget_set)).tolist()
    test_remain_set = Subset(test_remain_set, test_rm_idx)
    binary_test_set = ConcatDataset([test_forget_set, test_remain_set])
    binary_train_set = ConcatDataset([train_forget_set, train_remain_set])

    args.logger.info(f"Binary task loaded complete | Total train => fg:{len(train_forget_set)} rm:{len(train_remain_set)} | Total test => fg:{len(test_forget_set)} rm:{len(test_remain_set)} | Aug used: {args.aug_type}")

    
    binary_train_loader = DataLoader(dataset=binary_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    binary_test_loader = DataLoader(dataset=binary_test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return binary_train_loader, binary_test_loader



def generate_different_transforms(generate_aug_num, degree, args):
    '''
    TODO: make different types of augmentation
    '''
    assert degree in ['soft', 'hard'] 
    assert generate_aug_num < 3, "Not Implemented enough augmentation"

    normalize = transforms.Normalize(mean=args.mean, std=args.std)
    group_transforms_container = []

    if degree == "soft":
        for _ in range(generate_aug_num):
            single_transforms_container = []
            if np.random.random() > 0.4:
                single_transforms_container.append(transforms.RandomResizedCrop(size=args.image_size))
            if np.random.random() > 0.35:
                degree = int(np.random.random() * 45)
                single_transforms_container.append(transforms.RandomRotation((-degree, degree)))
            if np.random.random() > 0.4:
                prob = np.random.random()
                single_transforms_container.append(transforms.RandomHorizontalFlip(p=prob))
            if np.random.random() > 0.7:
                degree = int(np.random.random() * 30)
                single_transforms_container.append(transforms.RandomAffine((-degree, degree), (0.1, 0.1)))

            if np.random.random() > 0.2:
                single_transforms_container.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2))
            
            single_transforms_container.extend([transforms.ToTensor(), normalize])
            group_transforms_container.append(transforms.Compose(single_transforms_container))
    else:
        raise NotImplementedError

    return group_transforms_container
       

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

    return mean, std, image_size, num_classes

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

    # features = features.cpu().numpy()
    # np_sig_values = np.linalg.svd(features, full_matrices=False)[1]
    # np_sig_values = np_sig_values / np_sig_values.sum() + eps

    # np_features_rank = np.exp((np_sig_values * -np.log(np_sig_values)).sum())
    # breakpoint()

    return features_rank

def getCov(X):
    X_mean=X-np.mean(X,axis=0,keepdims=True)
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1) 
    return cov

def getHscore(f,Z):
    #Z=np.argmax(Z, axis=1)
    Covf=getCov(f)
    
    alphabetZ=list(set(Z))
    g=np.zeros_like(f)
    for z in alphabetZ:
        Ef_z=np.mean(f[Z==z, :], axis=0)
        g[Z==z]=Ef_z
    
    Covg=getCov(g)

    score=np.trace(np.dot(np.linalg.pinv(Covf,rcond=1e-15), Covg))
    return score

def get_multi_ranks(features_dict, args):
    labels = []
    for key in features_dict.keys():
        labels.extend([key]*features_dict[key].shape[0])
    
    cat = np.concatenate([features_dict[key] for key in features_dict.keys()], axis=0)
    zero_mean_total = (cat - cat.mean())
    labels = np.array(labels)
    score = getHscore(zero_mean_total, labels)

    return {
        'h_score': score,
    }
        
def get_specific_ranks(forget_matrices, remain_matrices, args):
    assert forget_matrices.shape[1] == remain_matrices.shape[1], "embeddings length should be matched"
    assert forget_matrices.shape[0] == args.dataset_length and remain_matrices.shape[0] == args.dataset_length

    dataset_length = args.dataset_length
    labels = torch.cat((torch.zeros(dataset_length), torch.ones(dataset_length))).numpy()
    labels_fg, labels_rm = torch.zeros(dataset_length).numpy(), torch.zeros(dataset_length).numpy() 

    fm = forget_matrices.cpu().numpy()
    rm = remain_matrices.cpu().numpy()

    zero_fm = (fm - fm.mean())
    zero_rm = (rm - rm.mean())

    fg_score = getHscore(zero_fm, labels_fg)
    rm_score = getHscore(zero_rm, labels_rm)

    cat = np.concatenate((fm, rm),axis=0)
    zero_mean_total = (cat - cat.mean())
    score = getHscore(zero_mean_total, labels)

    # zero_mean_fm = (forget_matrices - torch.mean(forget_matrices, dim=1).reshape(-1,1)).cpu().numpy() 
    # zero_mean_rm = (remain_matrices - torch.mean(remain_matrices, dim=1).reshape(-1,1)).cpu().numpy()

    # zero_mean_total = np.concatenate((zero_mean_fm, zero_mean_rm), axis=0)
    # score = getHscore(zero_mean_total, labels)

    # fg_N = forget_matrics.shape[0]
    # forget_matrics = forget_matrics.cpu().numpy()
    # fg_sim = cosine_similarity(forget_matrics)
    # low_fg_indices = np.tril_indices(fg_N, k=-1)
    # unique_fg_sim = fg_sim[low_fg_indices] 
    # fg_sim_sum = unique_fg_sim.mean()

    # rm_N = remain_matrices.shape[0]
    # remain_matrices = remain_matrices.cpu().numpy()
    # rm_sim = cosine_similarity(remain_matrices)
    # low_rm_indices = np.tril_indices(rm_N, k=-1)
    # unique_rm_sim = rm_sim[low_rm_indices] 
    # rm_sim_sum = unique_rm_sim.mean()

    # together_matrices = np.concatenate((forget_matrics, remain_matrices), axis=0)
    # intra_sim = cosine_similarity(together_matrices)
    

    fg_only_rank = calculate_feature_rank(forget_matrices) 
    rm_only_rank = calculate_feature_rank(remain_matrices)

    together_matrices = torch.cat((forget_matrices, remain_matrices))
    together_rank = calculate_feature_rank(together_matrices) #합집합 

    rank_overlap = fg_only_rank.item() + rm_only_rank.item() - together_rank.item() 
    pure_fg_rank = fg_only_rank.item() - rank_overlap
    pure_rm_rank = rm_only_rank.item() - rank_overlap


    args.logger.info(f"FG rank: {round(fg_only_rank.item())} | RM rank: {round(rm_only_rank.item())} | Together_rank: {round(together_rank.item())} | Rank_overlaps: {round(rank_overlap)} | IOU: {round(rank_overlap/together_rank.item(),4)*100}%")

    result_dict = {
        'fg_rank': fg_only_rank.item(),
        'rm_rank': rm_only_rank.item(),
        'together_rank': together_rank.item(),
        'h_score': score,
    }

    return result_dict


def calculate_inter_intra_similarity(forget_matrics, remain_matrices, args):
    pass
# itr, extractor, probe, binary_train_loader, 'train', rank_check, args

def metric_eval(model, loader, args):
    model.eval()
    embeddings = defaultdict(list)
    for idx, (inputs, labels) in enumerate(tqdm(loader)):
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        with torch.no_grad():
            _ , all_embeddings = model(inputs, get_all_features=True)
        for label_idx, label in enumerate(labels):
            embeddings[label.item()].append(all_embeddings['final'][label_idx].cpu().numpy())
    
    for key in tqdm(embeddings.keys()):
        # cat all embeddings in dim 0
        embeddings[key] = np.stack(embeddings[key], axis=0)
    
    return embeddings



def binary_eval(current_itr, model, probe, loader, type, rank_check, args):
    data_loader = loader
    assert type in ['train', 'test'], 'Type of evaluation should be noted'
    loss_fn = nn.BCEWithLogitsLoss()
    data_length = 0
    model_length = 0 
    correct = 0
    probe.eval()
    forget_indices = list(range(args.class_idx, args.class_idx + args.class_idx_unlearn))

    fg_features_container = []
    rm_features_container = []    
    fg_idx = rm_idx = 0

    for idx, (inputs, labels) in enumerate(tqdm(data_loader)):
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        #remove
        # remain_indices = torch.where(labels != 4)[0]
        # labels = labels[remain_indices]

        mask = torch.zeros_like(labels, dtype=torch.bool).to(args.device)
        for cls in forget_indices:
            mask = mask | (labels == cls)
        
        labels[mask] = 0
        labels[~mask] = 1

        one_hot_labels = F.one_hot(labels, num_classes=2).float() 
        with torch.no_grad():
            _ , all_embeddings = model(inputs, get_all_features=True)

        if idx == 0 and rank_check is not None:
            for ckpt in rank_check:
                assert ckpt == 'final', "Not implemented for higher dimensional features yet"
                col_length = all_embeddings[ckpt].shape[1]
                fg_features_container.append(torch.zeros((args.dataset_length, col_length)).to(args.device)) 
                rm_features_container.append(torch.zeros((args.dataset_length, col_length)).to(args.device)) 

        if rank_check is not None:
            fg_indices, rm_indices = torch.where(one_hot_labels[:,0]==1)[0], torch.where(one_hot_labels[:,1]==1)[0]
      
            for idx, ckpt in enumerate(rank_check):
                assert ckpt == 'final', "Not implemented for higher dimensional features yet"
                if len(fg_indices) > 0:
                    fg_features =  all_embeddings[ckpt][fg_indices]
                    fg_features_container[idx][fg_idx : fg_idx+len(fg_indices)] = fg_features
                    fg_idx += len(fg_indices)
                if len(rm_indices) > 0:
                    rm_features = all_embeddings[ckpt][rm_indices]
                    rm_features_container[idx][rm_idx : rm_idx+len(rm_indices)] = rm_features
                    rm_idx += len(rm_indices)
    
                
        final_embedding = all_embeddings['final']
        output = probe(final_embedding)
        if type == 'train':
            batch_loss = loss_fn(output, one_hot_labels)
            data_length += batch_loss.detach().cpu().numpy() * inputs.shape[0]

        pred = output.data.max(1)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()

    acc = (correct/len(loader.dataset))*100 
    
    if type == 'train':
        model_length = float(probe.kl_divergence().detach().cpu().numpy())
        args.logger.info(f"Train Acc: {acc:.3f}% | Model Code: {round(model_length)} bits, Data code: {round(data_length)} in itr: {current_itr}")
    else:
        args.logger.info(f"Test Acc: {acc:.3f}% | Model Code: {round(model_length)} bits, Data code: {round(data_length)} in itr: {current_itr}")


    result_dict =  {
        'acc': acc,
        'model_length': model_length,
        'data_length': data_length,
        'fg_container': fg_features_container,
        'rm_container': rm_features_container
    }


    return result_dict