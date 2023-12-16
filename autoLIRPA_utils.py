from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import *

import numpy as np
import torch 

def get_device():
    """
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


@torch.no_grad()
def compute_bounds(model, input_l, input_u, x):
    device = get_device()
    ptb = PerturbationLpNorm(norm=np.inf, x_L=input_l, x_U=input_u)
    x = BoundedTensor(x, ptb).to(device)
    l, u = model.compute_bounds(x=(x,), method='forward')
    return l, u