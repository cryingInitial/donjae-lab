import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
import copy
import pickle


@torch.no_grad()
def calculate_rank(model, loaders, name, args=None):
    train_loader, test_loader = loaders['train_test_loader'], loaders['test_loader']
    fg_train_loader, rm_train_loader = loaders['train_forget_test_loader'], loaders['train_remain_test_loader']
    fg_test_loader, rm_test_loader = loaders['test_forget_loader'], loaders['test_remain_loader']
    
    model.eval()
    eta = 1e-8  
    R = []
    for loader in [rm_train_loader, rm_test_loader, fg_train_loader, fg_test_loader, train_loader, test_loader]:
        batch_size = loader.batch_size
        Z = torch.zeros(len(loader.dataset), model.linear.weight.shape[1]).to(args.device)

        idx = 0
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            _, embeddings = model(inputs, get_embeddings=True)

            start = idx
            end  = start + embeddings.shape[0]
            Z[start:end] = embeddings
            idx = end

        _, S, _ = torch.linalg.svd(Z, full_matrices=False)
        
        S = (S / torch.sum(S)) + eta
        rank = torch.exp(torch.sum(-S.log() * S))
        R.append(rank)

    args.logger.info(f"************** Rank Calculation *************") 
    args.logger.info(f"{name} model  | retain rank| train: {R[0]:.2f}, test: {R[1]:.2f}")
    args.logger.info(f"{name} model  | forget rank| train: {R[2]:.2f}, test: {R[3]:.2f}")
    args.logger.info(f"{name} model  | total rank| train: {R[4]:.2f}, test: {R[5]:.2f}")
    args.logger.info(f"*********************************************") 

    return R

@torch.no_grad()
def calculate_damaged_fraction(gold_model, unlearn_model, loaders, args=None):
    '''
    calculate how damaged the unlearn model after training with pseudo forget dataset
    '''
    # fg_train_loader, fg_test_loader = loaders['train_forget_test_loader'], loaders['test_forget_loader']
    fg_dataset = loaders['train_forget_set']
    rt_train_loader, rt_test_loader = loaders['train_remain_test_loader'], loaders['test_remain_loader']

    gold_model.eval()
    unlearn_model.eval()

    #Note that data already on GPU in loader
    gold_pseudo_loader = generate_pseudo_forget_loader(gold_model, fg_dataset, args)
    unlearn_pseudo_loader = generate_pseudo_forget_loader(unlearn_model, fg_dataset, args)

    gold_before_train_acc, gold_before_test_acc =  eval(gold_model, rt_train_loader, args), eval(gold_model, rt_test_loader, args)
    unlearn_before_train_acc, unlearn_before_test_acc =  eval(unlearn_model, rt_train_loader, args), eval(unlearn_model, rt_test_loader, args)

    # opt_name = args.optimizer
    method_name = args.method
    lr = 0.001 if method_name == 'finetune' else 0.0001
    epch = 1

    gold_model = copy.deepcopy(gold_model)
    unlearn_model = copy.deepcopy(unlearn_model)
    gold_optim = torch.optim.SGD(gold_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # unlearn_optim = torch.optim.Adam(unlearn_model.parameters(), lr=lr) if method_name == 'finetune' else torch.optim.SGD(unlearn_model.parameters(), lr=lr) 
    unlearn_optim = torch.optim.SGD(gold_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    retrained_gold_model = retrain_model(gold_model, gold_pseudo_loader, gold_optim, epch)
    retrained_unlearned_model = retrain_model(unlearn_model, unlearn_pseudo_loader, unlearn_optim, epch)

    retrained_gold_model.eval()
    retrained_unlearned_model.eval()

    gold_after_train_acc, gold_after_test_acc =  eval(retrained_gold_model, rt_train_loader, args), eval(retrained_gold_model, rt_test_loader, args)
    unlearn_after_train_acc, unlearn_after_test_acc =  eval(retrained_unlearned_model, rt_train_loader, args), eval(retrained_unlearned_model, rt_test_loader, args)

    gold_train_damage_rate, gold_test_damage_rate = -((gold_after_train_acc-gold_before_train_acc)/gold_before_train_acc)*100, -((gold_after_test_acc-gold_before_test_acc)/gold_before_test_acc)*100
    unlearn_train_damage_rate, unlearn_test_damage_rate = -((unlearn_after_train_acc-unlearn_before_train_acc)/unlearn_before_train_acc)*100, -((unlearn_after_test_acc-unlearn_before_test_acc)/unlearn_before_test_acc)*100

    args.logger.info(f"************** Damage Fraction **************") 
    args.logger.info(f"Gold    result| train: {gold_train_damage_rate:.2f}%, test: {gold_test_damage_rate:.2f}%")
    args.logger.info(f"Unlearn result| train: {unlearn_train_damage_rate:.2f}%, test: {unlearn_test_damage_rate:.2f}%")
    args.logger.info(f"*********************************************") 


@torch.no_grad() 
def generate_pseudo_forget_loader(model, dataset, args=None):

    pseudo_data_list = []
    for idx, data in enumerate(dataset):
        img, _ = data
        img = img.to(args.device)
        img = img.unsqueeze(0)

        prediction = model(img)
        soft_label = F.softmax(prediction)
        
        pseudo_data = (img[0], soft_label)
        pseudo_data_list.append(pseudo_data)
    
    pseudo_dataset = SimpleDataset(pseudo_data_list)
    batch_size = 128 if args == None else args.batch_size
    forget_loader = DataLoader(dataset=pseudo_dataset, batch_size=batch_size, shuffle=True)

    return forget_loader

@torch.enable_grad()
def retrain_model(unlearned_model, dataloader, optimizer, epoch=1): 
    '''
    retrain the model after 'unlearning'. 
    epoch is a hyperparameter 
    '''
    unlearned_model.train()
    criterion = nn.CrossEntropyLoss()

    for i in range(epoch):
        for idx, (img, label) in enumerate(dataloader):
            label = label.squeeze(1)
            prediction = unlearned_model(img)
            loss = criterion(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return unlearned_model


class SimpleDataset(Dataset):

    def __init__(self, datalist):
        self.data = datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@torch.no_grad()
def calculate_activation_dist_with_divergence(orig_model, gold_model, unlearn_model, loaders, args=None):
    #Activation distance from "Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks Using an Incompetent Teacher" AAAI-23
    #code from: https://github.com/vikram2000b/bad-teaching-unlearning/blob/main/metrics.py
    '''
    calculate average l2 distance of prediction probabilities(logits) in forget set(original ver.)
    Here we calculate on both forget train and test set
    '''
    orig_model.eval()
    gold_model.eval()
    unlearn_model.eval()

    fg_train_loader = loaders['train_forget_test_loader']
    rt_train_loader = loaders['train_remain_test_loader']
    fg_test_loader = loaders['test_forget_loader']
    rt_test_loader = loaders['test_remain_loader']

    loader_pack = [fg_train_loader, fg_test_loader, rt_train_loader, rt_test_loader]

    dist_orig_result, dist_unlearn_result = [], []
    div_orig_result, div_unlearn_result= [], []
    
    for loader in loader_pack:
        #original, unlearn
        distances_orig = [] 
        distances_unlearn = []

        orig_prob = []
        gold_prob = []
        unlearn_prob = []

        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(args.device)
            orig_outputs = orig_model(inputs)
            gold_outputs = gold_model(inputs)
            unlearn_outputs = unlearn_model(inputs)

            diff_orig = torch.sqrt(torch.sum(torch.square(F.softmax(gold_outputs, dim = 1) - F.softmax(orig_outputs, dim = 1)), axis = 1))
            diff_orig = diff_orig.detach().cpu()
            distances_orig.append(diff_orig)
    
            diff_unlearn= torch.sqrt(torch.sum(torch.square(F.softmax(gold_outputs, dim = 1) - F.softmax(unlearn_outputs, dim = 1)), axis = 1))
            diff_unlearn = diff_unlearn.detach().cpu()
            distances_unlearn.append(diff_unlearn)

            orig_prob.append(F.softmax(orig_outputs + 1e-7, dim=1))
            gold_prob.append(F.softmax(gold_outputs + 1e-7, dim=1))
            unlearn_prob.append(F.softmax(unlearn_outputs + 1e-7, dim=1))

        orig_prob = torch.cat(orig_prob, axis = 0).cpu()
        gold_prob = torch.cat(gold_prob, axis = 0).cpu()
        unlearn_prob = torch.cat(unlearn_prob, axis = 0).cpu()
        js_div_un, js_std_un= JSDiv(gold_prob, unlearn_prob)
        js_div_orig, js_std_orig = JSDiv(gold_prob, orig_prob)
    
        distances_unlearn = torch.cat(distances_unlearn, axis = 0)
        distances_unlearn_std = ((distances_unlearn - distances_unlearn.mean())**2).mean()
        distances_orig = torch.cat(distances_orig, axis = 0)
        distances_orig_std = ((distances_unlearn - distances_unlearn.mean())**2).mean()

        dist_unlearn_result.append((distances_unlearn.mean(),distances_unlearn_std))
        dist_orig_result.append((distances_orig.mean(), distances_orig_std))
        div_unlearn_result.append((js_div_un, js_std_un))
        div_orig_result.append((js_div_orig, js_std_orig))


    for idx, (dist, div) in enumerate([(dist_orig_result, div_orig_result), (dist_unlearn_result, div_unlearn_result)]):
        idx = "orig   " if idx == 0 else 'unlearn'
        args.logger.info(f"Activate Distance :[Gold, {idx}] | Train_forget: {dist[0][0]:.3f} Std: {dist[0][1]:.3f} | Test_forget: {dist[1][0]:.3f} Std: {dist[1][1]:.3f} | Train_Remain: {dist[2][0]:.3f} Std: {dist[2][1]:.3f} | Test_Remain: {dist[3][0]:.3f} Std: {dist[3][1]:.3f}")
        args.logger.info(f"JS divergence     :[Gold, {idx}] | Train_forget: {div[0][0]:.3f} Std: {div[0][1]:.3f} | Test_forget: {div[1][0]:.3f} Std: {div[1][1]:.3f} | Train_Remain: {div[2][0]:.3f} Std: {div[2][1]:.3f} | Test_Remain: {div[3][0]:.3f} Std: {div[3][1]:.3f}")

def JSDiv(p, q):
    m = (p+q)/2
    batch_wise_kl = 0.5*F.kl_div(torch.log(p), m, reduction='batchmean') + 0.5*F.kl_div(torch.log(q), m, reduction='batchmean')
    logit_wise_kl = (0.5*F.kl_div(torch.log(p), m, reduction='none') + 0.5*F.kl_div(torch.log(q), m, reduction='none')).sum(axis=1)
    std = ((logit_wise_kl - batch_wise_kl)**2).mean() 
    return batch_wise_kl, std


def eval(model, loader, args, source=False):
    if loader is None: return 0
    total = 0; correct = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        outputs = model(inputs)
        predicted = torch.argmax(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    if source:
        return (correct, total)

    return 100. * correct / total 

def flops_easy_view(total_flops):
    flops = ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    i = 0
    while total_flops >= 1000:
        total_flops /= 1000
        i += 1
    return f"{total_flops:.2f} {flops[i]}FLOPS"

def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def collect_prob(data_loader, model):
    data_loader = torch.utils.data.DataLoader(
        data_loader.dataset, batch_size=1, shuffle=False
    )
    prob = []
    with torch.no_grad():
        for batch in data_loader:
            data, _ = batch
            data = data.cuda()
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)

def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    X_r = (
        torch.cat([entropy(retain_prob), entropy(test_prob)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r

# Selective Synaptic Dampening (SSD) - https://arxiv.org/abs/2308.07707
# https://github.com/if-loops/selective-synaptic-dampening/blob/main/src/metrics.py#L54
def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, model
    )
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_r, Y_r) # fit to retain(1) and test(0) data
    results = clf.predict(X_f)
    return results.mean()


def evaluate_summary(model, retrain_model, result_model, statistics, loaders, args):
    # R_retrain = calculate_rank(result_model, loaders, 'unlearn', args)
    
    # calculate_rank(model, loaders, 'original', args)
    # calculate_rank(result_model, loaders, 'unlearn', args)
    # evaluate_model(retrain_model, loaders, "Retrain Model", args)
    # E_retrain = evaluate_model(result_model, loaders, "Unlearned Model", args)

    file_path = args.pickle_path
    # write_arrays_to_pickle(file_path, R_retrain, E_retrain)

    # evaluate_model(retrain_model, loaders, "Retrain Model", args)
    # evaluate_model(result_model, loaders, "Unlearned Model", args)
    # calculate_activation_dist_with_divergence(model,retrain_model, result_model, loaders, args)
    # print(f"Total FLOPS: {flops_easy_view(statistics.total_flops)}, Elapsed time: {statistics.elapsed_time:.2f} sec")

@torch.no_grad()
def evaluate_model(model, loaders, model_name, args):
    model.eval()
    # acc_train = eval(model, loaders['train_test_loader'], args)
    # acc_test = eval(model, loaders['test_loader'], args)
    acc_train_forget = eval(model, loaders['train_forget_test_loader'], args)
    acc_train_remain = eval(model, loaders['train_remain_test_loader'], args)
    acc_test_forget = eval(model, loaders['test_forget_loader'], args)
    acc_test_remain = eval(model, loaders['test_remain_loader'], args)
    # mia = get_membership_attack_prob(
    #     loaders['train_remain_test_loader'], loaders['train_forget_test_loader'], loaders['test_loader'], model,
    # ) * 100

    args.logger.info(f"{model_name} Train_forget: {acc_train_forget:.2f}, Train_remain: {acc_train_remain:.2f}")
    args.logger.info(f"{model_name} Test_forget: {acc_test_forget:.2f}, Test_remain: {acc_test_remain:.2f}")

    acc_train = acc_test = mia = 0 
    return [acc_train, acc_test, acc_train_forget, acc_train_remain, acc_test_forget, acc_test_remain, mia]

def write_arrays_to_pickle(file_path, array1, array2):
    with open(file_path, 'ab') as file:  # 'ab' mode to append in binary
        pickle.dump({"array1": array1, "array2": array2}, file_path)