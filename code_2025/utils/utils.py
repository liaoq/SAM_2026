import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def converter(indices, num_classes):
    '''
    Convert indices to probability vectors
    '''
    probvec = torch.zeros(len(indices), num_classes)
    probvec[range(len(indices)), indices] = 1.0
    return probvec