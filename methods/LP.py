import torch
import torch.nn as nn
import torch.nn.functional as F
from adv_generator import inf_generator
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from methods.method import Method
from eval import eval

class LP(Method):
    def set_hyperparameters(self):
        return super().set_hyperparameters()

    def unlearn(self, model, loaders, args):
        device = args.device

        if args.retain_ratio < 1:
            sub_train_set = torch.utils.data.Subset(self.train_remain_set, np.random.choice(len(self.train_remain_set), int(len(self.train_remain_set) * args.retain_ratio), replace=False))
            print(f"Number of samples in the subset: {len(sub_train_set)}, retain ratio: {args.retain_ratio}")
            self.train_remain_loader = torch.utils.data.DataLoader(sub_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        remain_data_gen = inf_generator(self.train_remain_loader)
        criterion = nn.CrossEntropyLoss()

        model.train()

        # freeze the model
        for param in model.named_parameters():
            # except for last layer freeze the model
            if "linear" not in param[0]:
                param[1].requires_grad = False
        

        model.linear = nn.Linear(512, args.num_classes).to(device)
        optimizer = self.get_optimizer(model)

        for itr in tqdm(range(1, args.iter_num+1)):
            x_remain, y_remain = remain_data_gen.__next__()

            x_remain, y_remain = x_remain.to(device), y_remain.to(device)
            logits_remain = model(x_remain)

            self.statistics.add_forward_flops(x_remain.size(0))
            
            ce_loss = criterion(logits_remain, y_remain)

            model.zero_grad()
            optimizer.zero_grad()

            ce_loss.backward()
            self.statistics.add_backward_flops(x_remain.size(0))
            optimizer.step()
            
            if itr % args.test_interval == 0: self.intermidiate_test(model)

        return model
