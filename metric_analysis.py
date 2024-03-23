import torch
import pickle
import numpy as np
import json

def main(path):

    train_acc = []
    model_code = []
    data_code = [] 
    fg_rank = []
    rm_rank = [] 
    together_rank = []
    overlapped_rank = [] 
    last_model_code = 0
    last_data_code = 0 

    with open(path, 'r') as f:
    # Load the data from the JSON file
        data = json.load(f)

    data[-1]['model_length'] = last_model_code
    data[-1]['data_length'] = last_data_code
    
    for idx, element in enumerate(data):
        train_acc.append(element['acc'])
        model_code.append(element['model_length'])
        data_code.append(element['data_length'])
        fg_rank.append(element['fg_rank'])
        rm_rank.append(element['rm_rank'])
        together_rank.append(element['together_rank'])
        overlapped_rank.append(element['fg_rank'] + element['rm_rank'] - element['together_rank'])
    
    total_cnt = idx+1

    acc = sum(train_acc) / total_cnt 
    


    breakpoint()
if __name__ == '__main__':
    path = "metric_logs/cifar10/binary_debugging_sample_4_10000_soft/0_data.json"
    main(path)