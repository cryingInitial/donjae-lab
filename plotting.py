import torch
import torch.nn as nn
import torch.nn.functional as F
from main import arg_parse
from data import get_dataset, get_dataloader, split_class_data, split_sample_data
from backbone import get_model
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import math

def get_unlearn_loader(trainset, testset, trainset_test, args):
    if args.test_mode == 'class': 
        forget_index, remain_index = split_class_data(trainset, args.class_idx, args.class_unlearn)
        test_forget_index, test_remain_index = split_class_data(testset, args.class_idx, math.inf)
        
        
        test_forget_set = torch.utils.data.Subset(testset, test_forget_index)
        test_remain_set = torch.utils.data.Subset(testset, test_remain_index)
        test_forget_loader = DataLoader(dataset=test_forget_set, batch_size=1, shuffle=False)
        test_remain_loader = DataLoader(dataset=test_remain_set, batch_size=1, shuffle=False)

    elif args.test_mode == 'sample': 
        forget_index, remain_index = split_sample_data(trainset, args.sample_unlearn_per_class, args.data_name)
        test_forget_set = None
        test_remain_set = testset
        test_forget_loader = None
        test_remain_loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)

    # extract only 10 samples from each class
    per_remain_index = []
    for i in range(9):
        per_remain_index.extend(list(range(5000*i, 5000*i+10)))
    train_forget_set = torch.utils.data.Subset(trainset, forget_index)
    train_remain_set = torch.utils.data.Subset(trainset, per_remain_index)
    train_forget_loader = DataLoader(dataset=train_forget_set, batch_size=1, shuffle=False)
    train_remain_loader = DataLoader(dataset=train_remain_set, batch_size=1, shuffle=False)

    train_forget_test_set = torch.utils.data.Subset(trainset_test, forget_index)
    train_remain_test_set = torch.utils.data.Subset(trainset_test, per_remain_index)
        
    train_forget_test_loader = DataLoader(dataset=train_forget_test_set, batch_size=1, shuffle=False)
    train_remain_test_loader = DataLoader(dataset=train_remain_test_set, batch_size=1, shuffle=False)

    return train_forget_set, train_remain_set, test_forget_set, test_remain_set, train_forget_test_set, train_remain_test_set, \
        train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, train_forget_test_loader, train_remain_test_loader
        
def main():
    name = 'hey'
    
    ft_models = [get_model('resnet18', num_classes=10, ckpt_path=f'./lp_models/300_{i}.pth').to('cuda') for i in range(7)]
    original_models = [get_model('resnet18', num_classes=10, ckpt_path=f'./checkpoints/checkpoint/ResNet18_cifar10_adam_seed{i}_ori.pth').to('cuda') for i in range(7)]
    rt_models = [get_model('resnet18', num_classes=10, ckpt_path=f'./checkpoints/checkpoint/ResNet18_cifar10_adam_seed{i}_retrain.pth').to('cuda') for i in range(7)]
    random_models = [get_model('ResNet18', num_classes=10).to('cuda') for i in range(7)]
    
    # make them eval mode
    for model in ft_models: model.eval()
    for model in original_models: model.eval()
    for model in rt_models: model.eval()

    args = arg_parse()

    trainset, testset, trainset_test, two_transform, num_cls = get_dataset(args)
    train_loader, test_loader, train_test_loader = get_dataloader(trainset, testset, trainset_test, args)
    train_forget_set, train_remain_set, test_forget_set, test_remain_set, train_forget_test_set, train_remain_test_set,\
        train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, train_forget_test_loader, train_remain_test_loader = get_unlearn_loader(trainset, testset, trainset_test, args)

    # draw logit graph for every sample in train_forget_test_loader
    for i, (inputs, targets) in enumerate(train_forget_test_loader):
        plt.figure()
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        with torch.no_grad():
            ft_logits = [model(inputs, get_embeddings=False) for model in ft_models]
            original_logits = [model(inputs, get_embeddings=False) for model in original_models]
            rt_logits = [model(inputs, get_embeddings=False) for model in rt_models]
            random_logits = [model(inputs, get_embeddings=False) for model in random_models]
        # diff = torch.sqrt(torch.sum(torch.square(F.softmax(logit1, dim = 1) - F.softmax(logit2, dim = 1)), axis = 1))
        # diff = torch.sqrt(torch.sum(torch.square(F.softmax(logit1) - F.softmax(logit2)), axis = 1))
        
        # trying to make it zero for better visualization class 4
        # for logit in original_logits: logit[0][4] = torch.min(logit[0])
        
        ft_logits_for_plot = [F.softmax(logit * 0.07, dim=1).cpu().numpy() for logit in ft_logits]
        original_logits_for_plot = [F.softmax(logit * 0.07, dim=1).cpu().numpy() for logit in original_logits]
        rt_logits_for_plot = [F.softmax(logit * 0.07, dim=1).cpu().numpy() for logit in rt_logits]
        random_logits_for_plot = [F.softmax(logit * 0.07, dim=1).cpu().numpy() for logit in random_logits]
        
        for j in range(7):
            # plt.plot(ft_logits_for_plot[j][0], color='blue', linestyle=':', alpha=0.3)
            plt.plot(original_logits_for_plot[j][0], color='blue', linestyle=':', alpha=0.3)
            # plt.plot(random_logits_for_plot[j][0], color='blue', linestyle=':', alpha=0.3)
            plt.plot(rt_logits_for_plot[j][0], color='red', linestyle=':', alpha=0.3)
            
        mean_ft_logits_for_plot = sum(ft_logits_for_plot) / len(ft_logits_for_plot)
        mean_original_logits_for_plot = sum(original_logits_for_plot) / len(original_logits_for_plot)
        mean_rt_logits_for_plot = sum(rt_logits_for_plot) / len(rt_logits_for_plot)
        mean_random_logits_for_plot = sum(random_logits_for_plot) / len(random_logits_for_plot)
        
        # plt.plot(mean_ft_logits_for_plot[0], color='blue', label='Linear Probing')
        plt.plot(mean_original_logits_for_plot[0], color='blue', label='Original')
        # plt.plot(mean_random_logits_for_plot[0], color='blue', label='Random')
        plt.plot(mean_rt_logits_for_plot[0], color='red', label='Retrain')
        
        plt.title(f"Logit of {i}th sample", fontsize=20)
        plt.xlabel('Class', fontsize=15)
        plt.ylabel('Logit Value', fontsize=15)
        plt.ylim(0.05, 0.4)
        plt.legend(fontsize=14)
        plt.show()
        plt.savefig(f'./plot/{i}.png')
        if i == 50: break

if __name__ == '__main__':
    main()
    
