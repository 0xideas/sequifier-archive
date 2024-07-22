import os
import subprocess

import numpy as np
import pandas as pd
import torch
from torch import tensor

PANDAS_TO_TORCH_TYPES = {"int64": torch.int64, "float64": torch.float32}


def construct_index_maps(id_maps, target_columns_index_map, map_to_id):
    index_map = {}
    if map_to_id is not None:
        for target_column in target_columns_index_map:
            map_ = (
                {v: k for k, v in id_maps[target_column].items()} if map_to_id else None
            )
            if isinstance(list(map_.values())[0], str):
                map_[0] = "unknown"
            else:
                map_[0] = np.min(map_.values()) - 1
            index_map[target_column] = map_
    return index_map


def read_data(path, read_format, columns=None):
    if read_format == "csv":
        return pd.read_csv(path, sep=",", decimal=".", index_col=False)
    if read_format == "parquet":
        return pd.read_parquet(path, columns=columns)


def write_data(data, path, write_format, **kwargs):
    if write_format == "csv":
        return data.to_csv(path, sep=",", decimal=".", index=False, **kwargs)
    if write_format == "parquet":
        return data.to_parquet(path)


def subset_to_selected_columns(data, selected_columns):
    column_filters = [
        (data["inputCol"].values == input_col) for input_col in selected_columns
    ]
    filter_ = np.logical_or.reduce(column_filters)
    return data.loc[filter_, :]


def numpy_to_pytorch(
    data, column_types, selected_columns, target_columns, seq_length, device, to_device
):
    if "target" in data:
        targets = {}
        for target_column in target_columns:
            target = tensor(
                data.query(f"inputCol=='{target_column}'")[
                    [str(c) for c in range(seq_length - 1, 0, -1)] + ["target"]
                ].values
            ).to(column_types[target_column])
            if to_device:
                target = target.to(device)
            targets[target_column] = target
    else:
        targets = None

    sequence = {}
    for col in selected_columns:
        f = data["inputCol"].values == col
        data_subset = data.loc[f, [str(c) for c in range(seq_length, 0, -1)]].values

        tens = tensor(data_subset).to(column_types[col])

        if to_device:
            tens = tens.to(device)

        sequence[col] = tens

    return (sequence, targets)


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


def normalize_path(path, project_path):
    project_path_normalized = (project_path + os.sep).replace(os.sep + os.sep, os.sep)
    path2 = os.path.join(project_path, path.replace(project_path_normalized, ""))
    return path2
