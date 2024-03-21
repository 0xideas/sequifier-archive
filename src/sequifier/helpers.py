import subprocess

import torch
from torch import tensor

PANDAS_TO_TORCH_TYPES = {"int64": torch.int64, "float64": torch.float32}


def numpy_to_pytorch(data, column_types, target_column, seq_length, device, to_device):

    sequence = {
        col: (
            tensor(
                data.query(f"inputCol=='{col}'")[
                    [str(c) for c in range(seq_length, 0, -1)]
                ].values
            ).to(column_types[col])
        )
        for col in column_types.keys()
    }

    if to_device:
        sequence = {col: tens.to(device) for col, tens in sequence.items()}

    if "target" in data:
        target = tensor(data.query(f"inputCol=='{target_column}'")["target"].values).to(
            column_types[target_column]
        )
        if to_device:
            target = target.to(device)
    else:
        target = None

    return (sequence, target)


class LogFile(object):
    def __init__(self, path, open_mode):
        self._file = open(path, open_mode)
        self._path = path

    def write(self, string):
        self._file.write(f"{string}\n")
        self._file.flush()
        n_lines = string.count("\n")
        output = subprocess.check_output(
            f"tail -{n_lines} {self._path}", shell=True, text=True
        )
        print(output)

    def close(self):
        self._file.close()
