from adv_generator import inf_generator
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from methods.method import Method
from eval import eval

class FT(Method):

    def unlearn(self, model, loaders, args):
        device = args.device

        if args.retain_ratio < 1:
            sub_train_set = torch.utils.data.Subset(self.train_remain_set, np.random.choice(len(self.train_remain_set), int(len(self.train_remain_set) * args.retain_ratio), replace=False))
            print(f"Number of samples in the subset: {len(sub_train_set)}, retain ratio: {args.retain_ratio}")
            self.train_remain_loader = torch.utils.data.DataLoader(sub_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        remain_data_gen = inf_generator(self.train_remain_loader)

        criterion = nn.CrossEntropyLoss()
        optimizer = self.get_optimizer(model)
        
        model.train()
        for itr in tqdm(range(1, args.iter_num+1)):
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

            if itr % args.test_interval == 0: self.intermidiate_test(model)
                
        return model