import os

import torch
from torch import tensor


def numpy_to_pytorch(data, seq_length, device):

    sequence = (
        tensor(data[[str(c) for c in range(seq_length, 0, -1)]].values)
        .to(torch.int64)
        .to(device)
    )

    if "target" in data:
        target = tensor(data["target"].values).to(torch.int64).to(device)
    else:
        target = None

    return (sequence, target)
