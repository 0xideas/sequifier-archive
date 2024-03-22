import subprocess

import numpy as np
import torch
from torch import tensor

PANDAS_TO_TORCH_TYPES = {"int64": torch.int64, "float64": torch.float32}


def subset_to_selected_columns(data, selected_columns):
    column_filters = [
        (data["inputCol"].values == input_col) for input_col in selected_columns
    ]
    filter_ = np.logical_or.reduce(column_filters)
    return data.loc[filter_, :]


def numpy_to_pytorch(data, column_types, target_column, seq_length, device, to_device):

    if "target" in data:
        target = tensor(data.query(f"inputCol=='{target_column}'")["target"].values).to(
            column_types[target_column]
        )
        if to_device:
            target = target.to(device)
    else:
        target = None

    sequence = {}
    for col in column_types.keys():
        f = data["inputCol"].values == col
        data_subset = data.loc[f, [str(c) for c in range(seq_length, 0, -1)]].values

        tens = tensor(data_subset).to(column_types[col])
        if to_device:
            tens = tens.to(device)

        sequence[col] = tens

    return (sequence, target)


class LogFile(object):
    def __init__(self, path, open_mode):
        self._file = open(path, open_mode)
        self._path = path

    def write(self, string):
        self._file.write(f"{string}\n")
        self._file.flush()
        print(string)

    def close(self):
        self._file.close()
