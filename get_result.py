import os
import numpy as np
from collections import defaultdict
from easydict import EasyDict as edict
import pprint

dataset = 'cifar100'

dirs = os.listdir(f'./logs/{dataset}')

# parse Unlearned Model Train: 
# Unlearned Model Train: 86.73, Train_forget: 13.50, Train_remain: 94.93
# Unlearned Model Test: 81.14, Test_forget: 6.70, Test_remain: 89.41
# Unlearned Model MIA: 0.583

def parser(log):
    result = {}
    for line in log:
        if 'Unlearned Model Train' in line:
            line = line.split(', ')
            result['train'] = float(line[0].split(': ')[1])
            result['train_forget'] = float(line[1].split(': ')[1])
            result['train_remain'] = float(line[2].split(': ')[1])

        if 'Unlearned Model Test' in line:
            line = line.split(', ')
            result['test'] = float(line[0].split(': ')[1])
            result['test_forget'] = float(line[1].split(': ')[1])
            result['test_remain'] = float(line[2].split(': ')[1])
        
        if 'Unlearned Model MIA' in line:
            result['mia'] = float(line.split(': ')[1])

# [INFO] Activate Distance :[Gold, orig] | Train_forget: 1.277| Test_forget: 1.101| Train_Remain: 0.018| Test_Remain: 0.207
# [INFO] JS divergence     :[Gold, orig] | Train_forget: 0.336 | Test_forget: 0.248 | Train_Remain: 0.002 | Test_Remain: 0.019
# [INFO] Activate Distance :[Gold, unlearn] | Train_forget: 0.609| Test_forget: 0.599| Train_Remain: 0.240| Test_Remain: 0.282
# [INFO] JS divergence     :[Gold, unlearn] | Train_forget: 0.050 | Test_forget: 0.047 | Train_Remain: 0.037 | Test_Remain: 0.027
        # if 'Activate Distance' in line:
        #     line = line.split('|')
        #     result['activate_distance_train_forget'] = float(line[1].split(': ')[1])
        #     result['activate_distance_test_forget'] = float(line[2].split(': ')[1])
        #     result['activate_distance_train_remain'] = float(line[3].split(': ')[1])
        #     result['activate_distance_test_remain'] = float(line[4].split(': ')[1])

        # if 'JS divergence' in line:
        #     line = line.split('|')
        #     result['js_divergence_train_forget'] = float(line[1].split(': ')[1])
        #     result['js_divergence_test_forget'] = float(line[2].split(': ')[1])
        #     result['js_divergence_train_remain'] = float(line[3].split(': ')[1])
        #     result['js_divergence_test_remain'] = float(line[4].split(': ')[1])

    return result

for dir in dirs:
    results = defaultdict(list)
    if dir.startswith('scrub_r_cifar100_class_4_500_1.0_sgd'):
        # breakpoint()
        try:
            for seed in range(3):
                log = open(f'./logs/{dataset}/{dir}/seed{seed}.log').readlines()
                try:
                    result = parser(log)
                    for key in result.keys():
                        results[key].append(result[key])
                except:
                    pass

            # make it mean / std
            for key in results.keys():
            # 소수점 2자리까지만 출력
                results[key] = (f"{round(sum(results[key]) / len(results[key]), 2)} / {round(np.std(results[key]), 2)}")

            print(dir, len(results))
            pprint.pprint(dict(results))
        except Exception as e:
            print(e)
            pass
        