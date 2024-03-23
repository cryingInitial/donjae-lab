from adv_generator import inf_generator
from tqdm import tqdm
import copy
import torch.nn.utils.prune as prune
import torch
from torch import nn
from methods.method import Method
from eval import eval
from copy import deepcopy
from methods.pretrain import Pretrain
from methods.FT import FT
from methods.neggrad import NegGrad, NegGradP
from methods.teacher import Teacher
from methods.boundary_expand import BoundaryExpand
from methods.boundary_shrink import BoundaryShrink
from methods.cfk import CF_k
from methods.euk import EU_k
from methods.ntk import NTK
from methods.scrub import SCRUB
from methods.scrub_r import SCRUB_R
from methods.LP import LP
from methods.LPFT import LPFT
from methods.lau import LAU
from util import Statistics

class SPARSE(Method):

    def omp(self, model, args):
        args.rate = 0.95
        args.random_prune = True

        dense_model = copy.deepcopy(model)

        if args.random_prune:
            print("random pruning")
            self.pruning_model_random(model, args.rate)
        else:
            print("L1 pruning")
            self.pruning_model(model, args.rate)

        self.check_sparsity(model)
        current_mask = self.extract_mask(model.state_dict())
        self.remove_prune(model)

        #Completely Pruned
        self.prune_model_custom(dense_model, current_mask) 

        return dense_model

    def run(self):
        self.set_hyperparameters()
        pruned_model = self.omp(self.model, self.args)
        self.statistics = Statistics(pruned_model, self.args)
        self.statistics.start_record()
        model = self.unlearn(pruned_model, self.loaders, self.args)
        self.statistics.end_record()

        return model, self.statistics


    def unlearn(self, pruned_model, loaders, args):

        args.unlearn_method = 'ft'
        unlearn_method = args.unlearn_method.lower()

        if unlearn_method == 'ft':
            result_model = FT(pruned_model, loaders, args).unlearn(pruned_model, loaders, args)
        elif unlearn_method == 'neggrad':
            result_model = NegGrad(pruned_model, loaders, args).unlearn(pruned_model, loaders, args)
        elif unlearn_method == 'neggradp':
            result_model = NegGradP(pruned_model, loaders, args).unlearn(pruned_model, loaders, args)
        elif unlearn_method == 'lp':
            result_model = LP(pruned_model, loaders, args).unlearn(pruned_model, loaders, args)
        elif unlearn_method == 'lpft':
            result_model = LPFT(pruned_model, loaders, args).unlearn(pruned_model, loaders, args)
        elif unlearn_method == 'teacher':
            result_model = Teacher(pruned_model, loaders, args).unlearn(pruned_model, loaders, args)
        elif unlearn_method == 'scrub':
            result_model = SCRUB(pruned_model, loaders, args).unlearn(pruned_model, loaders, args)
        elif unlearn_method == 'scrub_r':
            result_model = SCRUB_R(pruned_model, loaders, args).unlearn(pruned_model, loaders, args)
        elif unlearn_method == 'pgd':
            result_model = PGD(pruned_model, loaders, args).unlearn(pruned_model, loaders, args)
        else:
            raise NotImplementedError
                
        return result_model
    

    def check_sparsity(self, model):
        #code from https://github.com/OPTML-Group/Unlearn-Sparse/
        sum_list = 0
        zero_sum = 0

        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                sum_list = sum_list + float(m.weight.nelement())
                zero_sum = zero_sum + float(torch.sum(m.weight == 0))

        if zero_sum:
            remain_weight_ratie = 100 * (1 - zero_sum / sum_list)
            print("* remain weight ratio = ", f"{100 * (1 - zero_sum / sum_list):.3f}", "%")
        else:
            print("no weight for calculating sparsity")
            remain_weight_ratie = None

        return remain_weight_ratie

    def pruning_model_random(self, model, px):
        #code from https://github.com/OPTML-Group/Unlearn-Sparse/
        print("Apply Unstructured Random Pruning Globally (all conv layers)")
        parameters_to_prune = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                parameters_to_prune.append((m, "weight"))

        parameters_to_prune = tuple(parameters_to_prune)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=px,
        )

    def pruning_model(self, model, px):
        #code from https://github.com/OPTML-Group/Unlearn-Sparse/
        print("Apply Unstructured L1 Pruning Globally (all conv layers)")
        parameters_to_prune = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                parameters_to_prune.append((m, "weight"))

        parameters_to_prune = tuple(parameters_to_prune)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=px,
        )

    def extract_mask(self, model_dict):
        #code from https://github.com/OPTML-Group/Unlearn-Sparse/
        new_dict = {}
        for key in model_dict.keys():
            if "mask" in key:
                new_dict[key] = copy.deepcopy(model_dict[key])

        return new_dict
    
    def remove_prune(self, model):
        #code from https://github.com/OPTML-Group/Unlearn-Sparse/
        print("Remove hooks for multiplying masks (all conv layers)")
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                prune.remove(m, "weight")

    
    def prune_model_custom(self, model, mask_dict):
        #code from https://github.com/OPTML-Group/Unlearn-Sparse/
        print("Pruning with custom mask (all conv layers)")
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                mask_name = name + ".weight_mask"
                if mask_name in mask_dict.keys():
                    prune.CustomFromMask.apply(
                        m, "weight", mask=mask_dict[name + ".weight_mask"]
                    )
                else:
                    print("Can not find [{}] in mask_dict".format(mask_name))