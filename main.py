import os
import argparse
import numpy as np
import torch
import logging.config

from data import get_dataset, get_dataloader, get_unlearn_loader, get_forget_remain_loader
from backbone import get_model
from method import run_method
from trainer import train_and_save
from eval import evaluate_summary, evaluate_model
from util import report_sample_by_class

from collections import OrderedDict

def seed_torch(seed):
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def log(args):
    logging.config.fileConfig('logging.conf')
    args.logger = logging.getLogger()
    os.makedirs(f'./logs/{args.data_name}', exist_ok=True)
    if args.test_mode == 'class': args.method_model_name = f'{args.method}_{args.data_name}_{args.test_mode}_{args.class_idx}_{args.class_unlearn}_{args.unlearn_epochs}_{args.note}'
    elif args.test_mode == 'sample': args.method_model_name = f'{args.method}_{args.data_name}_{args.test_mode}_{args.sample_unlearn_per_class}_{args.unlearn_epochs}_{args.note}'
    os.makedirs(f'./logs/{args.data_name}/{args.method_model_name}', exist_ok=True)
    fileHandler = logging.FileHandler(f'./logs/{args.data_name}/{args.method_model_name}/seed{args.rnd_seed}.log', mode='w')
    args.logger.addHandler(fileHandler)

def arg_parse():
    parser = argparse.ArgumentParser("Boundary Unlearning")
    parser.add_argument('--rnd_seed', type=int, default=0, help='random seed') # 0, 1, 2
    parser.add_argument('--method', type=str, default='ft', help='unlearning method')
    parser.add_argument('--data_name', type=str, default='cifar10', help='dataset, mnist or cifar10')
    parser.add_argument('--model_name', type=str, default='ResNet18', help='model name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--test_mode', type=str, default='sample', choices=['sample', 'class'], help='unlearning mode')
    parser.add_argument('--unlearn_epochs', type=float, help='unlearning epochs', required=True)
    parser.add_argument('--debug_mode', action='store_true', help='debug mode, it takes more time but with more detailed information')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer')
    parser.add_argument('--save_result_model', action='store_true', help="save result model")
    parser.add_argument('--note', type=str, default='', help='note')
    parser.add_argument('--test_interval', type=int, default=-1, help='test interval')
    parser.add_argument('--retain_ratio', type=float, default=1, help='retain ratio')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')

    # model selection
    parser.add_argument('--unlearn_aug', action='store_true', help="unlearn with data augmentation")

    # class unlearning, test_mode=class
    parser.add_argument('--class_idx', type=int, default=0, help='class index to unlearn')
    parser.add_argument('--class_unlearn', type=int, default=100, help='number of unlearning samples')

    # sample unlearning, test_mode=sample
    parser.add_argument('--sample_unlearn_per_class', type=int, default=100, help='number of unlearning samples per class')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
    seed_torch(args.rnd_seed)

    log(args)
    args.logger.info(args)
    args.logger.info(f'Model Selection: unlearn_aug={args.unlearn_aug}')
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.logger.info(f"Device: {args.device}")
    
        
    trainset, testset, trainset_test, trainset_two_transform, num_cls = get_dataset(args)
    args.logger.info("Dataset")
    train_loader, test_loader, train_test_loader = get_dataloader(trainset, testset, trainset_test, args)
    args.logger.info("Dataloader")

    if args.method == "contrastive":
        train_two_forget_set, train_two_remain_set, train_two_forget_loader, train_two_remain_loader = get_forget_remain_loader(trainset_two_transform, args)
    else: train_two_forget_set, train_two_remain_set, train_two_forget_loader, train_two_remain_loader = None, None, None, None
    train_forget_set, train_remain_set, test_forget_set, test_remain_set, train_forget_test_set, train_remain_test_set,\
        train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, train_forget_test_loader, train_remain_test_loader = get_unlearn_loader(trainset, testset, trainset_test, args)
    args.logger.info("Unlearn Dataloader")
    
    # if True:
    #     args.logger.info(f"train_forget: {report_sample_by_class(train_forget_loader)}, train_remain: {report_sample_by_class(train_remain_loader)}")
    #     args.logger.info(f"test_forget: {report_sample_by_class(test_forget_loader)}, test_remain: {report_sample_by_class(test_remain_loader)}")

    model_name = f"{args.model_name}_{args.data_name}_{args.test_mode}"
    if args.test_mode == 'class': model_name += f"_{args.class_idx}_{args.class_unlearn}"
    elif args.test_mode == 'sample': model_name += f"_{args.sample_unlearn_per_class}"

    if args.data_name == 'cifar10':
        # args.iter_num = 2000
        args.iter_num = 20000
        if args.test_interval == -1: args.test_interval = 20000
    elif args.data_name == 'cifar100':
        args.iter_num = 2000
        if args.test_interval == -1: args.test_interval = 500
    elif args.data_name == 'imagenet':
        args,iter_num = 2000
        if args.test_interval == -1: args.test_interval = 2000
        
    else: raise ValueError('Invalid dataset name')

    # print(f'TRYING TO LOAD {model_name}_ori{model_suffix}.pth...')
    # if not os.path.exists(f'./checkpoints/{model_name}_ori{model_suffix}.pth') or args.method.lower() == 'pretrain':
    #     print("THERE IS NO ORIGINAL MODEL, TRAINING A NEW ONE...", model_name)
    #     model = get_model(args.model_name, num_classes=args.num_classes).to(args.device)
    #     if args.data_name == 'cifar100':
    #         # model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False).to(args.device)

    #         from torchvision import models
    #         from torchvision.models import ResNet18_Weights
    #         imagenet_pretrained_state_dict = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
    #         del imagenet_pretrained_state_dict['fc.weight']
    #         del imagenet_pretrained_state_dict['fc.bias']
    #         del imagenet_pretrained_state_dict['conv1.weight']
    #         new_state_dict = OrderedDict()
    #         for k, v in imagenet_pretrained_state_dict.items():
    #             if 'downsample' in k:
    #                 new_state_dict[k.replace('downsample', 'shortcut')] = v
    #             else:
    #                 new_state_dict[k] = v
    #         model.load_state_dict(new_state_dict, strict=False)
    #     train_and_save(model, train_loader, test_loader, model_name, args, mode='ori')

    if not os.path.exists(f'./checkpoints/{model_name}_retrain.pth') or args.method.lower() == 'pretrain':
        print("THERE IS NO RETRAINED MODEL, TRAINING A NEW ONE...", model_name)
        model = get_model(args.model_name, num_classes=args.num_classes).to(args.device)
        if args.data_name == 'cifar100':
            # model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False).to(args.device)

            from torchvision import models
            from torchvision.models import ResNet18_Weights
            imagenet_pretrained_state_dict = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
            del imagenet_pretrained_state_dict['fc.weight']
            del imagenet_pretrained_state_dict['fc.bias']
            del imagenet_pretrained_state_dict['conv1.weight']
            new_state_dict = OrderedDict()
            for k, v in imagenet_pretrained_state_dict.items():
                if 'downsample' in k:
                    new_state_dict[k.replace('downsample', 'shortcut')] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
        train_and_save(model, train_remain_loader, test_loader, model_name, args, mode='retrain')

    if args.method.lower() != 'pretrain':

        loaders = dict(trainset=trainset, testset=testset, train_forget_set=train_forget_set, train_remain_set=train_remain_set, test_forget_set=test_forget_set, test_remain_set=test_remain_set, \
            train_loader=train_loader, test_loader=test_loader, train_forget_loader=train_forget_loader, train_remain_loader=train_remain_loader, test_forget_loader=test_forget_loader, test_remain_loader=test_remain_loader, \
            trainset_test=trainset_test, train_forget_test_set=train_forget_test_set, train_remain_test_set=train_remain_test_set, train_forget_test_loader=train_forget_test_loader, train_remain_test_loader=train_remain_test_loader, train_test_loader=train_test_loader, \
            trainset_two_transform=trainset_two_transform, train_two_forget_set=train_two_forget_set, train_two_remain_set=train_two_remain_set, train_two_forget_loader=train_two_forget_loader, train_two_remain_loader=train_two_remain_loader
            )

        model = get_model(args.model_name, num_classes=args.num_classes, ckpt_path=f'./checkpoints/{args.model_name}_{args.data_name}_ori.pth').to(args.device)
        # original_retrain_model = get_model(args.model_name, num_classes=args.num_classes, ckpt_path=f'./checkpoints/{model_name}_retrain.pth').to(args.device)
        retrain_model = get_model(args.model_name, num_classes=args.num_classes, ckpt_path=f'./checkpoints/{model_name}_retrain.pth').to(args.device)
        result_model, statistics = run_method(model, retrain_model, loaders, args)

        torch.save(result_model, f'./checkpoints/{args.method_model_name}{args.rnd_seed}_second_trial_{args.iter_num}.pth')

        # if args.save_result_model:
        #     torch.save(result_model, f'./checkpoints/{args.method_model_name}{args.rnd_seed}.pth')
        evaluate_model(retrain_model, loaders, "Retrain Original Model", args)
        evaluate_model(result_model, loaders, "Unlearn Model", args)
        # evaluate_summary(model, retrain_model, result_model, statistics, loaders, args)