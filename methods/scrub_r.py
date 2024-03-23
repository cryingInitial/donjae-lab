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
from methods.scrub import SCRUB, DistillKL, adjust_learning_rate
from torch.utils.data import DataLoader, Subset

class SCRUB_R(SCRUB):
    # Kurmanji et al. Towards Unbounded Machine Unlearning. NeurIPS, 2023
    # code from https://github.com/meghdadk/SCRUB/blob/main/MIA_experiments.ipynb
    def __init__(self, model, loaders, args):
        super().__init__(model, loaders, args)

    def unlearn(self, model, loaders, args):

        train_forget_loader, valid_forget_loader = None, None
        if args.test_mode == 'class':
            #class unlearning. should split forget train loader
            #Original scrub made validation set as 20%. We do 10% for validation
            forget_set = loaders['train_forget_set']
            forget_dataset, forget_indices = forget_set.dataset, torch.tensor(forget_set.indices)
            rand_idx = torch.randperm(int(len(forget_indices)*0.1))
            validation_indices = forget_indices[rand_idx].tolist()
            train_indices = list(set(forget_indices.tolist()).difference(set(validation_indices)))

            train_forget_loader = DataLoader(Subset(forget_dataset, train_indices), batch_size=args.batch_size, shuffle=True)
            valid_forget_loader = DataLoader(Subset(forget_dataset, validation_indices), batch_size=args.batch_size, shuffle=True)
    
        else:
            raise NotImplementedError

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

        t1 = time.time()
        acc_rs = []
        acc_fs = []
        acc_vs = []
        acc_fvs = []
        
        # forget_loader = loaders['train_forget_loader']
        # retain_loader = loaders['train_remain_loader']
        train_forget_loader = torch.utils.data.DataLoader(train_forget_loader.dataset, batch_size=args.fg_bs, shuffle=True)
        valid_forget_loader = torch.utils.data.DataLoader(valid_forget_loader.dataset, batch_size=args.fg_bs, shuffle=True)
        retain_loader = torch.utils.data.DataLoader(loaders['train_remain_loader'].dataset, batch_size=args.rt_bs, shuffle=True)
        
        scrub_name = "checkpoints/scrub_{}_{}_seed{}_step".format(args.model_name, args.data_name, args.rnd_seed)
        for epoch in range(1, args.sgda_epochs + 1):
            lr = adjust_learning_rate(epoch, args, optimizer)

            maximize_loss = 0
            if epoch <= args.msteps:
                maximize_loss = self.train_distill(epoch, train_forget_loader, module_list, swa_model, criterion_list, optimizer, args, "maximize")
            train_acc, train_loss = self.train_distill(epoch, retain_loader, module_list, swa_model, criterion_list, optimizer, args, "minimize",)
            if epoch >= args.sstart:
                swa_model.update_parameters(model_s)
            torch.save(model_s.state_dict(), f'{scrub_name}{epoch}_{args.lr}.pt')

            print ("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss, train_acc))
            # evaluate each time at the end of the epoch
            model_s.eval()
            acc_test = eval(model_s, loaders['test_loader'], args)
            acc_train_remain = eval(model_s, retain_loader, args)
            acc_train_forget = eval(model_s, train_forget_loader, args)
            acc_test_forget = eval(model_s, valid_forget_loader, args)
            acc_rs.append(100-acc_train_remain)
            acc_fs.append(100-acc_train_forget)
            acc_vs.append(100-acc_test)
            acc_fvs.append(100-acc_test_forget)
            model_s.train()
            
        try:
            selected_idx, _ = min(enumerate(acc_fs), key=lambda x: abs(x[1]-acc_fvs[-1]))
        except:
            selected_idx = len(acc_fs) - 1
        print ("the selected index is {}".format(selected_idx+1))
        selected_model = "checkpoints/scrub_{}_{}_seed{}_step{}_{}.pt".format(args.model_name, args.data_name, args.rnd_seed, int(selected_idx+1), args.lr)
        model_s_final = copy.deepcopy(model_s)
        model_s.load_state_dict(torch.load(selected_model))
        
        return model_s
    
    