import os 

from torch import tensor
import torch

def numpy_to_pytorch(data, seq_length, device):
    return(tensor(data[[str(c) for c in range(seq_length, 0, -1)]].values).to(torch.int64).to(device), tensor(data["target"].values).to(torch.int64).to(device))

