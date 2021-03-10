import os 

from torch import tensor
import torch

def numpy_to_pytorch(data, seq_length, device):
    return(tensor(data[[str(c) for c in range(seq_length, 0, -1)]].values).to(torch.int64).to(device), tensor(data["target"].values).to(torch.int64).to(device))


def create_folder_if_not_exists(path):
    path_elements = path.split("/")
    n_elements = len(path_elements)
    subpaths = ["/".join(path_elements[:i]) for i in range(1, n_elements+1)]
    for subpath in subpaths:
        if len(subpath) > 0:
            exists = os.path.exists(subpath)
            if not exists:
                os.makedirs(subpath)