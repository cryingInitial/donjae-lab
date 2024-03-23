from adv_generator import inf_generator
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from methods.method import Method

class CF_k(Method):
    # Goel et al. Evaluating inexact unlearning requires revisiting forgetting. Arxiv 2022
    # code from https://github.com/meghdadk/SCRUB/blob/main/MIA_experiments.ipynb
    def __init__(self, model, loaders, args):
        super().__init__(model, loaders, args)

        self.lr_decay_epochs = [10,15,20]
        self.sgda_learning_rate = 0.01
        self.lr_decay_rate = 0.1 # not sure

    def unlearn(self, model, loaders, args):
        for param in model.parameters():
            param.requires_grad_(False)

        if 'allcnn' in args.model_name.lower():
            layers = [9]
            for k in layers:
                for param in model.features[k].parameters():
                    param.requires_grad_(True)

        elif "resnet" in args.model_name.lower():
            for param in model.layer4.parameters():
                param.requires_grad_(True)
        else:
            raise NotImplementedError
        
        train_remain_loader = loaders['train_remain_loader']
        device = args.device

        remain_data_gen = inf_generator(train_remain_loader)
        
        batches_per_epoch = len(train_remain_loader)
        unlearn_epochs = args.unlearn_epochs

        criterion = nn.CrossEntropyLoss()
        optimizer = self.get_optimizer(model)
        
        model.train()
        for itr in tqdm(range(int(unlearn_epochs * batches_per_epoch))): 
            if itr % batches_per_epoch == 0:
                self.adjust_learning_rate(int(itr//batches_per_epoch), optimizer)

            x_remain, y_remain = remain_data_gen.__next__()
            
            x_remain = x_remain.to(device)
            y_remain = y_remain.to(device)
            
            logits = model(x_remain)
            self.statistics.add_forward_flops(x_remain.size(0))
            
            ce_loss = criterion(logits, y_remain)
            
            model.zero_grad()
            optimizer.zero_grad()

            loss = ce_loss
            loss.backward()
            self.statistics.add_backward_flops(x_remain.size(0))
            optimizer.step()
            
        return model

    def adjust_learning_rate(self, epoch, optimizer):
        """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
        steps = np.sum(epoch > np.asarray(self.lr_decay_epochs))
        new_lr = self.sgda_learning_rate
        if steps > 0:
            new_lr = self.sgda_learning_rate * (self.lr_decay_rate ** steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        return new_lr