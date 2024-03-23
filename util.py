from collections import defaultdict
from time import time
import torch
from torchvision import transforms
from fvcore.nn import FlopCountAnalysis, flop_count_table

def report_sample_by_class(data_loader):
    if data_loader is None: return None
    size = defaultdict(int)
    for batch, label in data_loader:
        for l in label:
            size[l.item()] += 1
    return size
    # it is not efficient to iterate through the entire dataset, so we will only consider labels not images

def get_logits(logits):
    try: return logits.logits
    except: return logits
    
class Statistics:
    def __init__(self, model, args):
        self.model = model
        mock_sample = torch.randn(1, 3, args.image_size, args.image_size).to(args.device)
        self.flops = FlopCountAnalysis(model, mock_sample).total()

        self.start_time = 0
        self.elapsed_time = 0
        self.total_flops = 0

    def start_record(self):
        self.start_time = time()
        self.total_flops = 0

    def end_record(self):
        self.elapsed_time = time() - self.start_time

    def add_forward_flops(self, num_samples):
        self.total_flops += self.flops * num_samples
        
    def add_backward_flops(self, num_samples):
        self.total_flops += 2 * self.flops * num_samples

    def add_matrix_multiplication_flops(self, A, B):
        assert A.shape[1] == B.shape[0]
        m, n, p = A.shape[0], B.shape[1], B.shape[0]
        self.total_flops += m * n * (2 * p - 1)

    def add_flops_manual(self, flops):
        self.total_flops += flops

class TwoTransform():
    def __init__(self, transform=None, n_views=2):
        self.transform = transform
        self.n_views = n_views
    
    def __call__(self, x):
        return [self.transform(x) for _ in range(self.n_views)]
        
    
    