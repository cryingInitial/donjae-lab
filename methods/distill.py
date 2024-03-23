from adv_generator import inf_generator
from tqdm import tqdm
import torch
from torch import nn
from methods.method import Method
from eval import eval
from copy import deepcopy
import torch.nn.functional as F
from backbone import get_model
from util import get_logits
from dkd import dkd_loss

class Distill(Method):

    def set_hyperparameters(self):
        self.temperature = 7
        return super().set_hyperparameters()
    def unlearn(self, model, loaders, args):
        device = args.device
        args.class_idx_unlearn = 1 

        remain_data_gen = inf_generator(self.train_remain_loader)
        forget_data_gen = inf_generator(self.train_forget_loader)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = self.get_optimizer(model)
        # distill_model = self.retrain_model
        remain_teacher = deepcopy(model)
        forget_teacher = deepcopy(model)
        
        model.train()
        forget_teacher.train()
        
            
        for itr in tqdm(range(1, args.iter_num+1)):
            x_remain, y_remain = remain_data_gen.__next__()
            x_forget, y_forget = forget_data_gen.__next__()
            
            x_remain = x_remain.to(device)
            y_remain = y_remain.to(device)
            x_forget = x_forget.to(device)
            y_forget = y_forget.to(device)
            
            remain_logits = get_logits(model(x_remain))
            forget_logits = get_logits(model(x_forget))
            
            teacher_remain_logits = get_logits(remain_teacher(x_remain)).detach()
            teacher_forget_logits = get_logits(forget_teacher(x_forget)).detach()
            
            self.statistics.add_forward_flops(x_remain.size(0))
            # forget distillation from original model
            # make class 4 to be least value for each logit
            # teacher_forget_logits[:, 4] = torch.min(teacher_forget_logits, dim = 1)[0]
            teacher_forget_confidence = F.softmax(teacher_forget_logits, dim = 1)
            teacher_forget_logits[:, args.class_idx:args.class_idx+args.class_idx_unlearn] = -1e20
            teacher_remain_logits[:, args.class_idx:args.class_idx+args.class_idx_unlearn] = -1e20

            # 0303_remain_forget
            # distill_remain_loss = nn.KLDivLoss()(F.log_softmax(remain_logits / self.temperature, dim = 1), F.softmax(teacher_remain_logits / self.temperature, dim = 1)) * (self.temperature ** 2)
            # distill_remain_loss = dkd_loss(remain_logits, teacher_remain_logits, y_remain, 0.05, 0.95, self.temperature)

            # [FIRST TRIAL]
            # distill_forget_loss = nn.KLDivLoss()(F.log_softmax(forget_logits / self.temperature, dim = 1), F.softmax(teacher_forget_logits / self.temperature, dim = 1)) * (self.temperature ** 2)
            # [SECOND TRIAL]
            distill_forget_loss = dkd_loss(forget_logits, teacher_forget_logits, y_forget, 0.95, 0.05, self.temperature)

            ce_loss = criterion(remain_logits, y_remain)
            

            model.zero_grad()
            optimizer.zero_grad()
            # print(distill_forget_loss, distill_remain_loss, ce_loss)
            # loss = distill_forget_loss + distill_remain_loss
            # loss = distill_forget_loss + distill_remain_loss
            loss = distill_forget_loss
            
            loss.backward()
            self.statistics.add_backward_flops(x_remain.size(0))
            optimizer.step()
            
            if itr % args.test_interval == 0: self.intermidiate_test(model)                
        return model