import os

import torch
from torch import tensor


def numpy_to_pytorch(data, column_types, seq_length, device):

    sequence = {col: (
        tensor(data.query(f"input_col=='{col}'")[[str(c) for c in range(seq_length, 0, -1)]].values)
        .to(column_types[col])
        .to(device)
    ) for col in column_types.keys()}

    if "target" in data:
        target = tensor(data.query(f"input_col=='itemId'")["target"].values).to(torch.int64).to(device)
    else:
        target = None

    return (sequence, target)
