import torch
import numpy as np

SEED = 311551147

def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)  
    np.random.seed(SEED)
