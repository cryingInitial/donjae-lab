from methods.method import Method
from adv_generator import *
from copy import deepcopy
from tqdm import tqdm

class BoundaryShrink(Method):
    def set_hyperparameters(self):
        self.bound=.1
        self.lambda_=.7
        self.bias=-.5
        self.slope=5.
        self.extra_exp="weight_assign"


    def unlearn(self, model, loaders, args):
        
        random_start = False  # False if attack != "pgd" else True
        device = args.device

        test_model = deepcopy(model).to(device)
        unlearn_model = deepcopy(model).to(device)
        adv = LinfPGD(test_model, self.bound, False, random_start, device)

        train_forget_loader = loaders['train_forget_loader']
        forget_data_gen = inf_generator(train_forget_loader)
        batches_per_epoch = len(train_forget_loader)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=0.00001, momentum=0.9)

        num_hits = 0
        num_sum = 0
        nearest_label = []
        poison_epoch = 10

        for itr in tqdm(range(poison_epoch * batches_per_epoch)):

            x, y = forget_data_gen.__next__()
            x = x.to(device)
            y = y.to(device)
            test_model.eval()
            x_adv = adv.perturb(x, y, target_y=None, model=test_model, device=device)
            adv_logits = test_model(x_adv)
            pred_label = torch.argmax(adv_logits, dim=1)
            if itr >= (poison_epoch - 1) * batches_per_epoch:
                nearest_label.append(pred_label.tolist())
            num_hits += (y != pred_label).float().sum()
            num_sum += y.shape[0]

            # adv_train
            unlearn_model.train()
            unlearn_model.zero_grad()
            optimizer.zero_grad()

            ori_logits = unlearn_model(x)
            ori_loss = criterion(ori_logits, pred_label)

            # loss = ori_loss  # - KL_div
            if self.extra_exp == 'curv': #1
                ori_curv = self.curvature(model, x, y, h=0.9)[1]
                cur_curv = self.curvature(unlearn_model, x, y, h=0.9)[1]
                delta_curv = torch.norm(ori_curv - cur_curv, p=2)
                loss = ori_loss + self.lambda_ * delta_curv  # - KL_div
            elif self.extra_exp == 'weight_assign': #2
                weight = self.weight_assign(adv_logits, pred_label, bias=self.bias, slope=self.slope)
                ori_loss = (torch.nn.functional.cross_entropy(ori_logits, pred_label, reduction='none') * weight).mean()
                loss = ori_loss
            else:
                loss = ori_loss  # - KL_div

            loss.backward()
            optimizer.step()

        return unlearn_model

    def _find_z(self, model, inputs, targets, h):
        '''
        Finding the direction in the regularizer
        '''
        inputs.requires_grad_()
        outputs = model(inputs)
        loss_z = nn.CrossEntropyLoss()(model(inputs), targets)
        # loss_z.backward(torch.ones(targets.size()).to(self.device))
        loss_z.backward()
        grad = inputs.grad.data + 0.0
        norm_grad = grad.norm().item()
        z = torch.sign(grad).detach() + 0.  ###[64, 3, 32, 32]
        z = 1. * (h) * (z + 1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None] + 1e-7)  ###[64, 3, 32, 32]
        # zero_gradients(inputs)
        inputs.grad.zero_()
        model.zero_grad()

        return z, norm_grad


    def curvature(self, model, inputs, targets, h=3., lambda_=4):
        '''
        Regularizer term in CURE
        '''
        z, norm_grad = self._find_z(model, inputs, targets, h)

        inputs.requires_grad_()
        outputs_pos = model(inputs + z)
        outputs_orig = model(inputs)

        loss_pos = nn.CrossEntropyLoss()(outputs_pos, targets)
        loss_orig = nn.CrossEntropyLoss()(outputs_orig, targets)
        grad_diff = torch.autograd.grad((loss_pos - loss_orig), inputs, create_graph=True)[0]
        ##grad_outputs=torch.ones(targets.size()).to(self.device),
        # curv_profile = torch.sort(grad_diff.reshape(grad_diff.size(0), -1))[0]  ###[64, 3072]
        reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)  ###[64]
        # del grad_diff
        model.zero_grad()

        return torch.sum(lambda_ * reg) / float(inputs.size(0)), reg


    def PM(self, logit, target):#[128,10], [128]
        if logit.shape[1] == 10:
            eye = torch.eye(10).cuda() #[10, 10]
        else:
            eye = torch.eye(11).cuda()
        # tmp1 = eye[target]#转one-hot
        # tmp2 = logit.softmax(1)#【128，10】
        # tmp3 = tmp1*tmp2
        # tmp3 = tmp3.sum(1)
        probs_GT = (logit.softmax(1) * eye[target]).sum(1).detach()#[128]
        top2_probs = logit.softmax(1).topk(2, largest = True)#[128, 2]
        # tmp4 = (top2_probs[1] == target.view(-1,1)).float()#[128, 2]
        # tmp4 = tmp4.sum(1)#[128]
        # tmp4 = tmp4 == 1#[128]bool
        GT_in_top2_ind = (top2_probs[1] == target.view(-1,1)).float().sum(1) == 1#[128]bool
        probs_2nd = torch.where(GT_in_top2_ind, top2_probs[0].sum(1) - probs_GT, top2_probs[0][:,0]).detach()
        return  probs_2nd - probs_GT


    def weight_assign(self, logit, target, bias, slope):
        pm = self.PM(logit, target)
        reweight = ((pm + bias) * slope).sigmoid().detach()
        normalized_reweight = reweight * 3
        return normalized_reweight