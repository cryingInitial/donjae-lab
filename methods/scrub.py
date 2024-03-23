from adv_generator import inf_generator
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from methods.method import Method
import copy
import torch.optim as optim
import time
from eval import eval

class SCRUB(Method):
    # Kurmanji et al. Towards Unbounded Machine Unlearning. NeurIPS, 2023
    # code from https://github.com/meghdadk/SCRUB/ / Slightly modified for our framework

    def unlearn(self, model, loaders, args):

        args.gamma = 0.99
        args.alpha = 0.001
        args.beta = 0
        args.smoothing = 0.0
        args.clip = 0.2
        args.sstart = 10
        args.kd_T = 4
        args.distill = 'kd'

        args.sgda_learning_rate = args.lr
        args.lr_decay_epochs = [3,5,9]
        args.lr_decay_rate = 0.1
        args.sgda_weight_decay = 5e-4
        args.sgda_momentum = 0.9
        
        #Fill these arguments whenver adding new datasets
        args.msteps = None
        args.sgda_epochs = None
        args.fg_bs = None
        args.rt_bs = None

        if args.data_name == 'cifar10':
            if args.test_mode == 'class':
                args.msteps = 2
                args.sgda_epochs = 3
                args.fg_bs = 512
                args.rt_bs = 128
                args.lr = 0.0005
                
            else:
                args.msteps = 5
                args.sgda_epochs = 5
                args.fg_bs = 16
                args.rt_bs = 64
            
        else:
            args.msteps = 2
            args.sgda_epochs = 3
            args.fg_bs = 32
            args.rt_bs = 128



        model_t = copy.deepcopy(model)
        model_s = copy.deepcopy(model)

        #this is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
        #For SGDA smoothing
        beta = 0.1
        def avg_fn(averaged_model_parameter, model_parameter, num_averaged): return (
            1 - beta) * averaged_model_parameter + beta * model_parameter
        swa_model = torch.optim.swa_utils.AveragedModel(
            model_s, avg_fn=avg_fn)

        module_list = nn.ModuleList([])
        module_list.append(model_s)
        trainable_list = nn.ModuleList([])
        trainable_list.append(model_s)

        criterion_cls = nn.CrossEntropyLoss()
        criterion_div = DistillKL(args.kd_T)
        criterion_kd = DistillKL(args.kd_T)


        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)    # classification loss
        criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
        criterion_list.append(criterion_kd)     # other knowledge distillation loss

        # optimizer
        if args.optimizer == "sgd":
            optimizer = optim.SGD(trainable_list.parameters(),
                                lr=args.sgda_learning_rate,
                                momentum=args.sgda_momentum,
                                weight_decay=args.sgda_weight_decay)
        elif args.optimizer == "adam": 
            optimizer = optim.Adam(trainable_list.parameters(),
                                lr=args.sgda_learning_rate,
                                weight_decay=args.sgda_weight_decay)

        module_list.append(model_t)

        if torch.cuda.is_available():
            module_list.cuda()
            criterion_list.cuda()
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            swa_model.cuda()
        

        forget_loader = torch.utils.data.DataLoader(loaders['train_forget_loader'].dataset, batch_size=args.fg_bs, shuffle=True)
        retain_loader = torch.utils.data.DataLoader(loaders['train_remain_loader'].dataset, batch_size=args.rt_bs, shuffle=True)
        

        for epoch in range(1, args.sgda_epochs + 1):
            lr = adjust_learning_rate(epoch, args, optimizer)

            maximize_loss = 0
            if epoch <= args.msteps:
                maximize_loss = self.train_distill(epoch, forget_loader, module_list, swa_model, criterion_list, optimizer, args, "maximize")
            train_acc, train_loss = self.train_distill(epoch, retain_loader, module_list, swa_model, criterion_list, optimizer, args, "minimize",)
            if epoch >= args.sstart:
                swa_model.update_parameters(model_s)

            print ("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss, train_acc))

        final_model = module_list[0]
        return final_model
    
    
    def train_distill(self, epoch, train_loader, module_list, swa_model, criterion_list, optimizer, opt, split, quiet=False):
        """One epoch distillation"""
        # set modules as train()
        for module in module_list:
            module.train()
        # set teacher as eval()
        module_list[-1].eval()

        criterion_cls = criterion_list[0]
        criterion_div = criterion_list[1]
        criterion_kd = criterion_list[2]

        model_s = module_list[0]
        model_t = module_list[-1]

        # batch_time = AverageMeter()
        # data_time = AverageMeter()
        losses = AverageMeter()
        kd_losses = AverageMeter()
        top1 = AverageMeter()

        # end = time.time()
        for idx, data in enumerate(tqdm(train_loader)):
            if opt.distill in ['crd']:
                input, target, index, contrast_idx = data
            else:
                input, target = data
            # data_time.update(time.time() - end)

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                if opt.distill in ['crd']:
                    contrast_idx = contrast_idx.cuda()
                    index = index.cuda()

            # ===================forward=====================
            #feat_s, logit_s = model_s(input, is_feat=True, preact=False)
            logit_s = model_s(input)
            self.statistics.add_forward_flops(input.size(0))
            with torch.no_grad():
                #feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
                #feat_t = [f.detach() for f in feat_t]
                logit_t = model_t(input)
                self.statistics.add_forward_flops(input.size(0))

            # cls + kl div
            loss_cls = criterion_cls(logit_s, target)
            loss_div = criterion_div(logit_s, logit_t)

            # other kd beyond KL divergence
            loss_kd = 0
            # if opt.distill == 'kd':
            #     loss_kd = 0
            # elif opt.distill == 'hint':
            #     f_s = module_list[1](feat_s[opt.hint_layer])
            #     f_t = feat_t[opt.hint_layer]
            #     loss_kd = criterion_kd(f_s, f_t)
            # elif opt.distill == 'crd':
            #     f_s = feat_s[-1]
            #     f_t = feat_t[-1]
            #     loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
            # elif opt.distill == 'attention':
            #     g_s = feat_s[1:-1]
            #     g_t = feat_t[1:-1]
            #     loss_group = criterion_kd(g_s, g_t)
            #     loss_kd = sum(loss_group)
            # elif opt.distill == 'nst':
            #     g_s = feat_s[1:-1]
            #     g_t = feat_t[1:-1]
            #     loss_group = criterion_kd(g_s, g_t)
            #     loss_kd = sum(loss_group)
            # elif opt.distill == 'similarity':
            #     g_s = [feat_s[-2]]
            #     g_t = [feat_t[-2]]
            #     loss_group = criterion_kd(g_s, g_t)
            #     loss_kd = sum(loss_group)
            # elif opt.distill == 'rkd':
            #     f_s = feat_s[-1]
            #     f_t = feat_t[-1]
            #     loss_kd = criterion_kd(f_s, f_t)
            # elif opt.distill == 'pkt':
            #     f_s = feat_s[-1]
            #     f_t = feat_t[-1]
            #     loss_kd = criterion_kd(f_s, f_t)
            # elif opt.distill == 'kdsvd':
            #     g_s = feat_s[1:-1]
            #     g_t = feat_t[1:-1]
            #     loss_group = criterion_kd(g_s, g_t)
            #     loss_kd = sum(loss_group)
            # elif opt.distill == 'correlation':
            #     f_s = module_list[1](feat_s[-1])
            #     f_t = module_list[2](feat_t[-1])
            #     loss_kd = criterion_kd(f_s, f_t)
            # elif opt.distill == 'vid':
            #     g_s = feat_s[1:-1]
            #     g_t = feat_t[1:-1]
            #     loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            #     loss_kd = sum(loss_group)
            # else:
            #     raise NotImplementedError(opt.distill)

            if split == "minimize":
                loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
            elif split == "maximize":
                loss = -loss_div

            loss = loss + param_dist(model_s, swa_model, opt.smoothing)

            if split == "minimize" and not quiet:
                acc1, _ = accuracy(logit_s, target, topk=(1,1))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
            elif split == "maximize" and not quiet:
                kd_losses.update(loss.item(), input.size(0))
            # elif split == "linear" and not quiet:
            #     acc1, _ = accuracy(logit_s, target, topk=(1, 1))
            #     losses.update(loss.item(), input.size(0))
            #     top1.update(acc1[0], input.size(0))
            #     kd_losses.update(loss.item(), input.size(0))


            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            self.statistics.add_backward_flops(input.size(0))
            #nn.utils.clip_grad_value_(model_s.parameters(), clip)
            optimizer.step()

            # ===================meters=====================
            # batch_time.update(time.time() - end)
            # end = time.time()

        #     if not quiet:
        #         if split == "mainimize":
        #             if idx % opt.print_freq == 0:
        #                 print('Epoch: [{0}][{1}/{2}]\t'
        #                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #                     'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #                     'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #                     epoch, idx, len(train_loader), batch_time=batch_time,
        #                     data_time=data_time, loss=losses, top1=top1))
        #                 sys.stdout.flush()

        if split == "minimize":
            if not quiet:
                print(' * Acc@1 {top1.avg:.3f} '
                    .format(top1=top1))

            return top1.avg, losses.avg
        else:
            return kd_losses.avg
        
def adjust_learning_rate(epoch, args, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
    new_lr = args.sgda_learning_rate
    if steps > 0:
        new_lr = args.sgda_learning_rate * (args.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr

import torch.nn as nn
import torch.nn.functional as F

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def param_dist(model, swa_model, p):
    #This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist
