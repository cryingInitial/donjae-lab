import copy
import numpy as np
import torch
from torch import nn
from adv_generator import inf_generator
import tqdm
import time
from einops import rearrange
from methods.method import Method
import torch.nn.functional as F
from eval import eval

class LAU(Method):
    '''
    METHOD: Layer Attack Unlearning: Fast and Accurate Machine Unlearning via Layer Level Attack and Knowledge Distillation (AAAI 24)
    Conducted Experiments: CIFAR-10 (Single class), Fashion-MNIST (Single class), and VGGFace2 (10 individuals, single class) 
  
    PGD alpha:  0.4 to 0.6
    PGD epochs: Unknown 
    Unlearn Batch size: Unknown
    KD alpah: 0.5 best for CIFAR-10
    Softmax Temp: 4 best 
    Models: ResNet18, ResNet50, VGG16, ViT

    '''
    def unlearn(self, model, loaders, args):
        forget_loader = loaders['train_forget_test_loader']

        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.linear.parameters():
            param.requires_grad_(True)
        
        forget_data_gen = inf_generator(forget_loader)
        batches_per_epoch = len(forget_loader)
        unlearn_epochs = 3 #args.unlearn_epochs

        criterion = nn.CrossEntropyLoss()
        KLDiv = nn.KLDivLoss()
        optimizer = self.get_optimizer(model)
        
        T = 4
        alpha = 0.5
        model.train()
        teacher_linear = copy.deepcopy(model.linear).to(args.device)
        for itr in range(int(unlearn_epochs * batches_per_epoch)):
            x_forget, y_forget = forget_data_gen.__next__()
            x_forget = x_forget.to(args.device)
            y_forget = y_forget.to(args.device)

            if itr % batches_per_epoch == 0:
                with torch.no_grad():
                    if torch.all(torch.argmax(model(x_forget), dim=1) != y_forget):
                        #if all samples within a batch differes from true label, terminate
                        break
                    
                teacher_linear.load_state_dict(model.linear.state_dict())

            logit_s, x_embed = model(x_forget, get_embeddings = True)
            x_embed_adv = pgd_attack(teacher_linear, x_embed, y_forget, device=args.device, alpha=0.4, opt='original')

            with torch.no_grad():
                logit_t = teacher_linear(x_embed_adv)

            unlearned_indices = torch.argmax(logit_s, dim=1) == y_forget 
            labels = torch.argmax(logit_s, dim=1).detach()
            labels[unlearned_indices] = torch.argmax(logit_t[unlearned_indices], dim=1)
            L_ce = criterion(logit_s, labels)

            Z = torch.softmax(logit_s, dim=1).detach()
            Z[unlearned_indices] = torch.softmax(logit_t[unlearned_indices], dim=1)

            L_di = KLDiv(F.softmax(logit_s/T, dim=1).log(), torch.softmax(Z/T, dim=1))
            loss = (1-alpha)*L_ce + alpha*T*T*L_di

            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # if itr == 2000: break (For fixed iter experiments)

        return model

def pgd_attack(model, feats, labels, eps=0.3, alpha=0.4, iters=40,device=None, opt='original'):
    loss = nn.CrossEntropyLoss()
    ori_feats = feats.data

    pgd_model = copy.deepcopy(model)
    for i in range(iters):    
        feats.requires_grad = True
        outputs = pgd_model(feats)

        pgd_model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        if opt == 'l2':

            grad = feats.grad
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1) + 1e-10 
            grad = grad / grad_norm.view(grad_norm.shape[0], 1)
            adv_feats = feats + alpha * grad

            delta = adv_feats - ori_feats
            delta_norms = torch.norm(delta.view(grad_norm.shape[0], -1), p=2, dim=1)
            factor = eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1)
            feats = torch.clamp(ori_feats + delta, min=0, max=1).detach()

        else:
            adv_feats = feats + alpha*feats.grad.sign()
            delta = torch.clamp(adv_feats - ori_feats, min=-eps, max=eps)
            feats = torch.clamp(ori_feats + delta, min=0, max=1).detach()

    return feats


