import json
import math
import multiprocessing
import os
from argparse import ArgumentParser
from random import shuffle

import numpy as np
import pandas as pd

from sequifier.config.preprocess_config import load_preprocessor_config


class Preprocessor(object):
    def __init__(
        self,
        project_path,
        data_path,
        data,
        group_proportions,
        seq_length,
        seed,
        target_column,
        return_targets,
        max_rows=None,
        process_id=0,
    ):
        self.project_path = project_path
        self.seed = seed
        np.random.seed(seed)

        os.makedirs(os.path.join(project_path, "data"), exist_ok=True)

        self.data_name_root = os.path.split(data_path)[1].split(".")[0]
        self.split_paths = [
            os.path.join(
                self.project_path,
                "data",
                f"{self.data_name_root}-split{i}-{process_id}.csv",
            )
            for i in range(len(group_proportions))
        ]

        data_columns = [
            col for col in data.columns if col not in ["sequenceId", "itemPosition"]
        ]

        n_classes, id_maps = {}, {}
        float_data_columns = []
        for data_col in data_columns:
            dtype = str(data[data_col].dtype)
            if dtype in ["object", "int64"]:
                data, sup_id_map = self.replace_ids(data, column=data_col)
                id_maps[data_col] = dict(sup_id_map)
                n_classes[data_col] = len(np.unique(data[data_col])) + 1
            elif dtype in ["float64"]:
                float_data_columns.append(data_col)
            else:
                raise Exception(
                    f"Column {data_col} is of dtype {dtype}, which is not supported"
                )
        col_types = {col: str(data[col].dtype) for col in data_columns}
        self.export_metadata(id_maps, n_classes, col_types)

        data = data.sort_values(["sequenceId", "itemPosition"])
        sequence_ids = sorted(list(np.unique(data["sequenceId"])))
        for i, sequence_id in enumerate(sequence_ids):
            data_subset = data.loc[data["sequenceId"] == sequence_id, :]
            sequences = self.extract_sequences(
                data_subset, seq_length, data_columns, target_column, return_targets
            )

            splits = self.extract_data_subsets(sequences, group_proportions)
            splits = [self.cast_columns_to_string(data) for data in splits]

            for split_path, split in zip(self.split_paths, splits):
                write_mode = "w" if i == 0 else "a"
                header = i == 0
                split.to_csv(
                    split_path,
                    mode=write_mode,
                    header=header,
                    sep=",",
                    decimal=".",
                    index=None,
                )

        print(f"Written data to {self.split_paths}")

    def export_metadata(self, id_maps, n_classes, col_types):

        data_driven_config = {
            "n_classes": n_classes,
            "id_maps": id_maps,
            "split_paths": self.split_paths,
            "column_types": col_types,
        }
        os.makedirs(
            os.path.join(self.project_path, "configs", "ddconfigs"), exist_ok=True
        )

        with open(
            os.path.join(
                self.project_path, "configs", "ddconfigs", f"{self.data_name_root}.json"
            ),
            "w",
        ) as f:
            f.write(json.dumps(data_driven_config))

    @classmethod
    def cast_columns_to_string(cls, data):
        data.columns = [str(col) for col in data.columns]
        return data

    @classmethod
    def replace_ids(cls, data, column):
        ids = sorted(
            [int(x) if not isinstance(x, str) else x for x in np.unique(data[column])]
        )
        id_map = {id_: i + 1 for i, id_ in enumerate(ids)}
        data[column] = data[column].map(id_map)
        return (data, id_map)

    @classmethod
    def extract_subsequences(
        cls, in_seq, seq_length, data_columns, target_column, return_targets
    ):

        if return_targets:
            nseq = max(
                len(in_seq[target_column]) - seq_length - 1,
                min(1, len(in_seq[target_column])),
            )

            targets = [in_seq[target_column][i + seq_length] for i in range(nseq)]
        else:
            nseq = max(
                len(in_seq[target_column]) - seq_length,
                min(1, len(in_seq[target_column])),
            )
            targets = [np.nan for _ in range(nseq)]

        seqs = {}
        for data_col in data_columns:
            seqs[data_col] = [in_seq[data_col][i : i + seq_length] for i in range(nseq)]

        if len(seqs[target_column]) == 1:
            seqs = {
                col: [[0] * (seq_length - len(seqs[col][0])) + seqs[col][0]]
                for col in data_columns
            }

        return (seqs, targets)

    @classmethod
    def extract_sequences(
        cls, data, seq_length, data_columns, target_column, return_targets
    ):
        raw_sequences = (
            data.groupby("sequenceId")
            .agg({col: list for col in data_columns})
            .reset_index(drop=False)
        )

        rows = []
        for _, in_row in raw_sequences.iterrows():
            seqs, targets = cls.extract_subsequences(
                in_row[data_columns],
                seq_length,
                data_columns,
                target_column,
                return_targets,
            )
            for i, target in enumerate(targets):
                subsequence_id = i
                for data_col, data_col_seqs in seqs.items():
                    rows.append(
                        [in_row["sequenceId"]]
                        + [subsequence_id, data_col]
                        + data_col_seqs[i]
                        + [target if data_col == target_column else None]
                    )

        sequences = pd.DataFrame(
            rows,
            columns=["sequenceId", "subsequenceId", "inputCol"]
            + list(range(seq_length, 0, -1))
            + ["target"],
        )
        return sequences

    @classmethod
    def get_subset_groups(cls, sequence_data, groups, n_cols):
        n_cases = int(sequence_data.shape[0] / n_cols)
        subset_groups = [
            ([i] * math.floor(n_cases * size)) for i, size in enumerate(groups)
        ]
        subset_groups = [inner for outer in subset_groups for inner in outer]
        diff = n_cases - len(subset_groups)
        subset_groups = ([0] * diff) + subset_groups
        return subset_groups

    @classmethod
    def extract_data_subsets(cls, sequences, groups):
        assert abs(1.0 - np.sum(groups)) < 0.0000000000001, np.sum(groups)

        datasets = [[] for _ in range(len(groups))]
        n_cols = len(np.unique(sequences["inputCol"]))
        for _, sequence_data in sequences.groupby("sequenceId"):
            subset_groups = cls.get_subset_groups(sequence_data, groups, n_cols)
            assert len(subset_groups) * n_cols == sequence_data.shape[0]
            for i, group in enumerate(subset_groups):
                case_start = i * n_cols
                datasets[group].append(
                    sequence_data.iloc[case_start : case_start + n_cols, :]
                )

        return [pd.concat(dataset, axis=0) for dataset in datasets]


def preprocess_batch(config, process_id, batch):
    Preprocessor(
        **{
            "data": batch,
            "process_id": process_id,
            **config.dict(),
        }
    )


def get_batch_limits(data, n_batches):
    sequence_ids = data["sequenceId"].values

    new_sequence_id_indices = np.concatenate(
        [
            [0],
            np.where(
                np.concatenate([[False], sequence_ids[1:] != sequence_ids[:-1]], axis=0)
            )[0],
        ]
    )

    ideal_step = math.ceil(data.shape[0] / n_batches)
    ideal_limits = np.array(
        [ideal_step * m for m in range(n_batches)] + [data.shape[0]]
    )
    distances = [
        np.abs(new_sequence_id_indices - ideal_limit)
        for ideal_limit in ideal_limits[:-1]
    ]
    actual_limit_indices = [
        np.where(distance == np.min(distance))[0] for distance in distances
    ]
    actual_limits = [
        new_sequence_id_indices[limit_index[0]] for limit_index in actual_limit_indices
    ] + [data.shape[0]]
    return list(zip(actual_limits[:-1], actual_limits[1:]))


def combine_multiprocessing_outputs(n_splits, n_batches):
    for split in range(n_splits):
        files = [f"data/full-split{split}-{batch}.csv" for batch in range(n_batches)]
        command = " ".join(["csvstack"] + files + [f"> data/full-split{split}.csv"])
        os.system(command)
        delete_command = f"rm data/full-split{split}-*"
        os.system(delete_command)


def preprocess(args, args_config):
    config = load_preprocessor_config(args.config_path, args_config)
    data = pd.read_csv(config.data_path, sep=",", decimal=".", index_col=None)

    n_cores = multiprocessing.cpu_count()
    if config.max_rows is not None:
        data = data.head(int(config.max_rows))

    batch_limits = get_batch_limits(data, n_cores)
    batches = [
        (config, data.iloc[start:end, :])
        for start, end in batch_limits
        if (end - start) > 0
    ]
    batches = [(config, i, batch) for i, (config, batch) in enumerate(batches)]

    n_cores_used = len(batches)
    with multiprocessing.Pool(processes=n_cores_used) as pool:
        pool.starmap(preprocess_batch, batches)

    combine_multiprocessing_outputs(len(config.group_proportions), n_cores_used)

    print("Preprocessing complete")
