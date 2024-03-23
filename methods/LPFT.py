from methods.method import Method
from adv_generator import inf_generator

import torch
from torch import nn
from tqdm import tqdm
from eval import eval

class LPFT(Method):
    def unlearn(self, model, loaders, args):
        
        device = args.device
        remain_data_gen = inf_generator(self.train_remain_loader)

        criterion = nn.CrossEntropyLoss()
        optimizer_ft = self.get_optimizer(model)
        if self.args.optimizer == 'adam':
            optimizer_lp = torch.optim.Adam(model.parameters(), lr=self.args.lr * 20)
        elif self.args.optimizer == 'sgd':
            optimizer_lp = torch.optim.SGD(model.parameters(), lr=self.args.lr * 20, momentum=.9, weight_decay=5e-4)
        else:
            raise ValueError(f"Invalid optimizer: {self.args.optimizer}")
        
        optimizer_bundle = [optimizer_ft, optimizer_ft, optimizer_ft, optimizer_ft, optimizer_lp]
        
        model.train()
        for i in range(len(optimizer_bundle)):
            if i > 0:
                for param in model.named_parameters():
                    if f'layer{i}' in param[0]:
                        param[1].requires_grad = False
                        print(param[0] + " is frozen")
            
            print(f"# of unfrozen parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            if i == 0: epoch = args.iter_num // 2
            else: epoch = args.iter_num // 8
            for itr in tqdm(range(int(epoch))):
                x_remain, y_remain = remain_data_gen.__next__()
                x_remain, y_remain = x_remain.to(device), y_remain.to(device)
                logits = model(x_remain)
                self.statistics.add_forward_flops(x_remain.size(0))
                ce_loss = criterion(logits, y_remain)
                model.zero_grad()
                optimizer_bundle[i].zero_grad()
                ce_loss.backward()
                self.statistics.add_backward_flops(x_remain.size(0))
                optimizer_bundle[i].step()

            self.intermidiate_test(model)

        return model