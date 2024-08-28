import json
import math
import multiprocessing
import os
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sequifier.config.preprocess_config import load_preprocessor_config
from sequifier.helpers import read_data, write_data


def preprocess(args, args_config):
    config_path = (
        args.config_path if args.config_path is not None else "configs/preprocess.yaml"
    )
    config = load_preprocessor_config(config_path, args_config)
    Preprocessor(**config.dict())
    print("Preprocessing complete")


class Preprocessor(object):
    def __init__(
        self,
        project_path,
        data_path,
        read_format,
        write_format,
        selected_columns,
        return_targets,
        target_columns,
        group_proportions,
        seq_length,
        max_rows,
        seed,
        n_cores,
    ):
        self.project_path = project_path
        self.seed = seed
        np.random.seed(seed)

        os.makedirs(os.path.join(project_path, "data"), exist_ok=True)
        temp_path = os.path.join(self.project_path, "data", "temp")
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.makedirs(temp_path)

        data = read_data(data_path, read_format, columns=selected_columns)

        if selected_columns is not None:
            selected_columns_filtered = [
                col
                for col in selected_columns
                if col not in ["sequenceId", "itemPosition"]
            ]
            data = data[["sequenceId", "itemPosition"] + selected_columns_filtered]

        if max_rows is not None:
            data = data.head(int(max_rows))

        self.data_name_root = os.path.split(data_path)[1].split(".")[0]
        self.split_paths = [
            os.path.join(
                self.project_path,
                "data",
                f"{self.data_name_root}-split{i}.{write_format}",
            )
            for i in range(len(group_proportions))
        ]

        data_columns = [
            col for col in data.columns if col not in ["sequenceId", "itemPosition"]
        ]

        n_classes, id_maps = {}, {}
        min_max_values = {}
        float_data_columns = []
        for data_col in data_columns:
            dtype = str(data[data_col].dtype)
            if dtype in ["object", "int64"]:
                data, sup_id_map = replace_ids(data, column=data_col)
                id_maps[data_col] = dict(sup_id_map)
                n_classes[data_col] = len(np.unique(data[data_col])) + 1
            elif dtype in ["float64"]:
                min_ = np.min(data[data_col].values)
                max_ = np.max(data[data_col].values)
                data[data_col] = [
                    (((v - min_) / (max_ - min_)) * 1.6) - 0.8 for v in data[data_col]
                ]
                min_max_values[data_col] = {"min": min_, "max": max_}
                float_data_columns.append(data_col)
            else:
                raise Exception(
                    f"Column {data_col} is of dtype {dtype}, which is not supported"
                )
        col_types = {col: str(data[col].dtype) for col in data_columns}
        self.export_metadata(id_maps, n_classes, col_types, min_max_values)

        data = data.sort_values(["sequenceId", "itemPosition"])

        n_cores = n_cores if n_cores is not None else multiprocessing.cpu_count()

        batch_limits = get_batch_limits(data, n_cores)
        batches = [
            (
                data.iloc[start:end, :],
                self.split_paths,
                seq_length,
                data_columns,
                target_columns,
                return_targets,
                group_proportions,
                write_format,
            )
            for start, end in batch_limits
            if (end - start) > 0
        ]
        batches = [(i, *v) for i, v in enumerate(batches)]

        n_cores_used = len(batches)

        with multiprocessing.Pool(processes=n_cores_used) as pool:
            pool.starmap(preprocess_batch, batches)

        combine_multiprocessing_outputs(
            project_path,
            len(group_proportions),
            n_cores_used,
            f"{self.data_name_root}",
            write_format,
        )

        delete_path = os.path.join(project_path, "data", "temp")
        assert len(delete_path) > 9
        delete_command = f"rm -rf {delete_path}*"
        os.system(delete_command)

    def export_metadata(self, id_maps, n_classes, col_types, min_max_values):
        data_driven_config = {
            "n_classes": n_classes,
            "id_maps": id_maps,
            "split_paths": self.split_paths,
            "column_types": col_types,
            "min_max_values": min_max_values,
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


def replace_ids(data, column):
    ids = sorted(
        [int(x) if not isinstance(x, str) else x for x in np.unique(data[column])]
    )
    id_map = {id_: i + 1 for i, id_ in enumerate(ids)}
    data[column] = data[column].map(id_map)
    return (data, id_map)


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


def preprocess_batch(
    process_id,
    batch,
    split_paths,
    seq_length,
    data_columns,
    target_columns,
    return_targets,
    group_proportions,
    write_format,
):
    sequence_ids = sorted(list(np.unique(batch["sequenceId"])))
    written_files = {i: [] for i, _ in enumerate(split_paths)}
    for i, sequence_id in enumerate(sequence_ids):
        data_subset = batch.loc[batch["sequenceId"] == sequence_id, :]
        sequences = extract_sequences(
            data_subset, seq_length, data_columns, target_columns, return_targets
        )

        splits = extract_data_subsets(sequences, group_proportions)
        splits = {group: cast_columns_to_string(data) for group, data in splits.items()}

        for split_path, (group, split) in zip(split_paths, splits.items()):
            split_path_batch_seq = split_path.replace(
                f".{write_format}", f"-{process_id}-{i}.{write_format}"
            )
            split_path_batch_seq = insert_top_folder(split_path_batch_seq, "temp")

            if write_format == "csv":
                write_data(split, split_path_batch_seq, "csv", mode="w", header=True)
            if write_format == "parquet":
                write_data(split, split_path_batch_seq, "parquet")

            written_files[group] = written_files[group] + [split_path_batch_seq]

    for j, split_path in enumerate(split_paths):
        out_path = split_path.replace(
            f".{write_format}", f"-{process_id}.{write_format}"
        )
        out_path = insert_top_folder(out_path, "temp")

        if write_format == "csv":
            command = " ".join(["csvstack"] + written_files[j] + [f"> {out_path}"])
            os.system(command)

        if write_format == "parquet":
            combine_parquet_files(written_files[j], out_path)


def extract_sequences(data, seq_length, data_columns, target_columns, return_targets):
    raw_sequences = (
        data.groupby("sequenceId")
        .agg({col: list for col in data_columns})
        .reset_index(drop=False)
    )

    rows = []
    for _, in_row in raw_sequences.iterrows():
        seqs, targets = extract_subsequences(
            in_row[data_columns],
            seq_length,
            data_columns,
            target_columns,
            return_targets,
        )
        for i, target_dict in enumerate(targets):
            subsequence_id = i
            for data_col, data_col_seqs in seqs.items():
                rows.append(
                    [in_row["sequenceId"]]
                    + [subsequence_id, data_col]
                    + data_col_seqs[i]
                    + [target_dict.get(data_col, None)]
                )

    sequences = pd.DataFrame(
        rows,
        columns=["sequenceId", "subsequenceId", "inputCol"]
        + list(range(seq_length, 0, -1))
        + ["target"],
    )
    return sequences


def extract_subsequences(
    in_seq, seq_length, data_columns, target_columns, return_targets
):
    nseq = max(
        len(in_seq[target_columns[0]]) - seq_length - 1,  # any column will do
        min(1, len(in_seq[target_columns[0]])),  # any column will do
    )
    if nseq == 1:  # any column will do
        in_seq = {
            col: ([0] * (seq_length - len(in_seq[col]) + int(return_targets)))
            + in_seq[col]
            for col in data_columns
        }

    if return_targets:
        targets = [
            {
                target_column: in_seq[target_column][i + seq_length]
                for target_column in target_columns
            }
            for i in range(nseq)
        ]

    else:
        targets = [
            {target_column: np.nan for target_column in target_columns}
            for _ in range(nseq)
        ]

    seqs = {}
    for data_col in data_columns:
        seqs[data_col] = [in_seq[data_col][i : i + seq_length] for i in range(nseq)]

    return (seqs, targets)


def insert_top_folder(path, folder_name):
    components = os.path.split(path)
    new_components = list(components[:-1]) + [folder_name] + [components[-1]]
    return os.path.join(*new_components)


def extract_data_subsets(sequences, groups):
    assert abs(1.0 - np.sum(groups)) < 0.0000000000001, np.sum(groups)

    datasets = {i: [] for i in range(len(groups))}
    n_cols = len(np.unique(sequences["inputCol"]))
    for _, sequence_data in sequences.groupby("sequenceId"):
        subset_groups = get_subset_groups(sequence_data, groups, n_cols)
        assert len(subset_groups) * n_cols == sequence_data.shape[0]
        for i, group in enumerate(subset_groups):
            case_start = i * n_cols
            datasets[group].append(
                sequence_data.iloc[case_start : case_start + n_cols, :]
            )

    data_subset = {
        group: pd.concat(dataset, axis=0)
        for group, dataset in datasets.items()
        if len(dataset)
    }

    return data_subset


def get_subset_groups(sequence_data, groups, n_cols):
    n_cases = int(sequence_data.shape[0] / n_cols)
    subset_groups = [
        ([i] * math.floor(n_cases * size)) for i, size in enumerate(groups)
    ]
    subset_groups = [inner for outer in subset_groups for inner in outer]
    diff = n_cases - len(subset_groups)
    subset_groups = ([0] * diff) + subset_groups
    return subset_groups


def cast_columns_to_string(data):
    data.columns = [str(col) for col in data.columns]
    return data


def combine_multiprocessing_outputs(
    project_path, n_splits, n_batches, dataset_name, write_format
):
    for split in range(n_splits):
        out_path = os.path.join(
            project_path, "data", f"{dataset_name}-split{split}.{write_format}"
        )

        files = [
            os.path.join(
                project_path,
                "data",
                "temp",
                f"{dataset_name}-split{split}-{batch}.{write_format}",
            )
            for batch in range(n_batches)
        ]
        if write_format == "csv":
            command = " ".join(["csvstack"] + files + [f"> {out_path}"])
            os.system(command)
        if write_format == "parquet":
            combine_parquet_files(files, out_path)


def combine_parquet_files(files, out_path):
    schema = pq.ParquetFile(files[0]).schema_arrow
    with pq.ParquetWriter(out_path, schema=schema, compression="snappy") as writer:
        for file in files:
            writer.write_table(pq.read_table(file, schema=schema))
