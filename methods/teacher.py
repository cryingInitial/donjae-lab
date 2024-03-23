import torch
from adv_generator import inf_generator
from methods.method import Method  
from copy import deepcopy 
from backbone import get_model
from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

# Paper: https://arxiv.org/pdf/2205.08096.pdf
# Code: https://github.com/vikram2000b/bad-teaching-unlearning
class Teacher(Method):
    def set_hyperparameters(self):
        self.kl_temperature = 1.0

    def unlearn(self, model, loaders, args):
        self.set_hyperparameters()
        train_forget_set, train_remain_set = self.train_forget_set, self.train_remain_set
        # print transform of train_forget_set
        # print(train_forget_set.transform, train_remain_set.transform)
        # only use 30% of train_forget_set
        train_forget_set_30 = torch.utils.data.Subset(train_forget_set, np.random.choice(len(train_forget_set), int(len(train_forget_set) * 0.3), replace=False))
        unlearning_data = UnLearningData(train_forget_set_30, train_remain_set)
        unlearn_loader = torch.utils.data.DataLoader(unlearning_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        unlearn_data_gen = inf_generator(unlearn_loader)
        
        incompetent_teacher = get_model(name=args.model_name, num_classes=args.num_classes).to(args.device)
        compotent_teacher = deepcopy(model).to(args.device)
        incompetent_teacher.eval()
        compotent_teacher.eval()

        optimizer = self.get_optimizer(model)

        for itr in tqdm(range(1, args.iter_num+1)):
            self.unlearning_step(model, incompetent_teacher, compotent_teacher, unlearn_data_gen.__next__(), optimizer, args.device, self.kl_temperature)
        
        return model

    def UnlearnerLoss(self, output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
        labels = torch.unsqueeze(labels, dim = 1)
        
        f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
        u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

        # label 0: remain sample, label 1: forget sample
        overall_teacher_out = labels * u_teacher_out + (1-labels)*f_teacher_out
        student_out = F.log_softmax(output / KL_temperature, dim=1)
        return F.kl_div(student_out, overall_teacher_out)
    
    def unlearning_step(self, model, unlearning_teacher, full_trained_teacher, batch, optimizer, device, KL_temperature):
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        output = model(x)
        self.statistics.add_forward_flops(x.size(0)*2)
        optimizer.zero_grad()
        loss = self.UnlearnerLoss(output = output, labels=y, full_teacher_logits=full_teacher_logits, 
                unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=KL_temperature)
        loss.backward()
        self.statistics.add_backward_flops(x.size(0))
        optimizer.step()
    
class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len
    
    def __getitem__(self, index):
        if(index < self.forget_len):
            x = self.forget_data[index][0]
            y = 1
            return x,y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x,y