from adv_generator import inf_generator
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from methods.method import Method
import copy
from collections import OrderedDict
from backbone import get_model

class NTK(Method):
    # Golatkar et al. Forgetting outside the box: Scrubbing deep networks of information accessible from input-output observations. ECCV 2020
    # Code from https://github.com/AdityaGolatkar/SelectiveForgetting/blob/master/Forgetting.ipynb
    def __init__(self, model, loaders, args):
        super().__init__(model, loaders, args)
        self.model_init = get_model(args.model_name, num_classes=args.num_classes).to(args.device)

    def unlearn(self, model, loaders, args):
        args.weight_decay = 0.0005

        retain_loader = loaders['train_remain_loader']
        forget_loader = loaders['train_forget_loader']
        num_total = len(retain_loader.dataset) + len(forget_loader.dataset)
        num_to_retain = len(retain_loader.dataset)
        args.logger.info(f'total train data:{num_total}')
        args.logger.info(f'retain train data:{num_to_retain}')
        #### Scrubbing Direction
        # w_complete = np.load('NTK_data/w_complete.npy')
        # w_retain = np.load('NTK_data/w_retain.npy')
        args.lossfn = 'ce'

        # w_complete
        G_r,f0_minus_y_r = delta_w_utils(copy.deepcopy(model),retain_loader,args,'complete')
        args.logger.info('complete retain dataset')
        np.save('NTK_data/G_r.npy',G_r)
        np.save('NTK_data/f0_minus_y_r.npy',f0_minus_y_r)
        G_f,f0_minus_y_f = delta_w_utils(copy.deepcopy(model),forget_loader,args,'retain') 
        np.save('NTK_data/G_f.npy',G_f)
        np.save('NTK_data/f0_minus_y_f.npy',f0_minus_y_f)
        args.logger.info('complete forget dataset')
        
        G = np.concatenate([G_r,G_f],axis=1)

        f0_minus_y = np.concatenate([f0_minus_y_r,f0_minus_y_f])
        theta = G.transpose().dot(G) + num_total*args.weight_decay*np.eye(G.shape[1])
        theta_inv = np.linalg.inv(theta)
        w_complete = -G.dot(theta_inv.dot(f0_minus_y))


        theta_r = G_r.transpose().dot(G_r) + num_to_retain*args.weight_decay*np.eye(G_r.shape[1])
        theta_r_inv = np.linalg.inv(theta_r)
        w_retain = -G_r.dot(theta_r_inv.dot(f0_minus_y_r))

        delta_w = (w_retain-w_complete).squeeze()

        m_pred_error = vectorize_params(model)-vectorize_params(self.model_init)-w_retain.squeeze()
        args.logger.info(f"Delta w -------: {np.linalg.norm(delta_w)}")

        inner = np.inner(delta_w/np.linalg.norm(delta_w),m_pred_error/np.linalg.norm(m_pred_error))
        args.logger.info(f"Inner Product--: {inner}")

        if inner<0:
            angle = np.arccos(inner)-np.pi/2
            args.logger.info(f"Angle----------:  {angle}")

            predicted_norm=np.linalg.norm(delta_w) + 2*np.sin(angle)*np.linalg.norm(m_pred_error)
            args.logger.info(f"Pred Act Norm--:  {predicted_norm}")
        else:
            angle = np.arccos(inner) 
            args.logger.info(f"Angle----------:  {angle}")

            predicted_norm=np.linalg.norm(delta_w) + 2*np.cos(angle)*np.linalg.norm(m_pred_error)
            args.logger.info(f"Pred Act Norm--:  {predicted_norm}")

        predicted_scale=predicted_norm/np.linalg.norm(delta_w)
        predicted_scale
        args.logger.info(f"Predicted Scale:  {predicted_scale}")
        scale=predicted_scale
        direction = get_delta_w_dict(delta_w,model)

        model_scrub = copy.deepcopy(model)
        for k,p in model_scrub.named_parameters():
            p.data += (direction[k]*scale).to(args.device)

        return model_scrub

def get_delta_w_dict(delta_w,model):
    # Give normalized delta_w
    delta_w_dict = OrderedDict()
    params_visited = 0
    for k,p in model.named_parameters():
        num_params = np.prod(list(p.shape))
        update_params = delta_w[params_visited:params_visited+num_params]
        delta_w_dict[k] = torch.Tensor(update_params).view_as(p)
        params_visited+=num_params
    return delta_w_dict

def vectorize_params(model):
    param = []
    for p in model.parameters():
        param.append(p.data.view(-1).cpu().numpy())
    return np.concatenate(param)

def delta_w_utils(model_init,dataloader,args,name='complete'):
    model_init.eval()
    dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=1, shuffle=False)
    G_list = []
    f0_minus_y = []
    for idx, batch in enumerate(tqdm(dataloader)):#(tqdm(dataloader,leave=False)):
        batch = [tensor.to(next(model_init.parameters()).device) for tensor in batch]
        input, target = batch
        if 'mnist' in args.data_name:
            input = input.view(input.shape[0],-1)
        target = target.cpu().detach().numpy()
        output = model_init(input)
        G_sample=[]
        for cls in range(args.num_classes):
            grads = torch.autograd.grad(output[0,cls],model_init.parameters(),retain_graph=True)
            grads = np.concatenate([g.view(-1).cpu().numpy() for g in grads])
            G_sample.append(grads)
            G_list.append(grads)
        if args.lossfn=='mse':
            p = output.cpu().detach().numpy().transpose()
            #loss_hess = np.eye(len(p))
            target = 2*target-1
            f0_y_update = p-target
        elif args.lossfn=='ce':
            p = torch.nn.functional.softmax(output,dim=1).cpu().detach().numpy().transpose()
            p[target]-=1
            f0_y_update = copy.deepcopy(p)
        f0_minus_y.append(f0_y_update)
    return np.stack(G_list).transpose(),np.vstack(f0_minus_y)