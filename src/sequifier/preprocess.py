import json
import math
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from random import shuffle

from sequifier.config.preprocess_config import load_preprocessor_config


class Preprocessor(object):
    def __init__(
        self,
        project_path,
        data_path,
        group_proportions,
        seq_length,
        seed,
        max_rows=None,
    ):
        self.project_path = project_path
        self.seed = seed
        np.random.seed(seed)

        data = pd.read_csv(data_path, sep=",", decimal=".", index_col=None)

        if max_rows is not None:
            data = data.head(int(max_rows))

        os.makedirs(os.path.join(project_path, "data"), exist_ok=True)

        self.data_name_root = os.path.split(data_path)[1].split(".")[0]
        self.split_paths = [
            os.path.join(
                self.project_path, "data", f"{self.data_name_root}-split{i}.csv"
            )
            for i in range(len(group_proportions))
        ]

        supplementary_columns = [col for col in data.columns if col not in ["sequenceId", "itemId", "itemPosition"]]
        n_classes = {col: len(np.unique(data[col])) + 1 for col in ["itemId"] + supplementary_columns}

        data, id_map = self.replace_ids(data, column="itemId")

        id_maps = {"itemId": id_map}
        float_supplementary_columns = []
        for sup_col in supplementary_columns:
            dtype = str(data[sup_col].dtype)
            if dtype in ["object", "int64"]:
                data, sup_id_map = self.replace_ids(data, column=sup_col)
                id_maps[sup_col] = dict(sup_id_map)
            elif dtype in ["float64"]:
                float_supplementary_columns.append(sup_col)
            else:
                raise Exception(f"Column {sup_col} is of dtype {dtype}, which is not supported")
            
        sequences, col_types = self.extract_sequences(data, seq_length, supplementary_columns)


        self.splits = self.extract_data_subsets(sequences, group_proportions)
        self.splits = [self.cast_columns_to_string(data) for data in self.splits]
        self.export(id_maps, n_classes, col_types)

    def export(self, id_maps, n_classes, col_types):

        data_driven_config = {
            "n_classes": n_classes,
            "id_maps": id_maps,
            "split_paths": self.split_paths,
            "column_types": col_types
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

        for split_path, split in zip(self.split_paths, self.splits):
            split.to_csv(split_path, sep=",", decimal=".", index=None)
            print(f"Written data to {split_path}")

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
    def extract_subsequences(cls, in_seq, seq_length, supplementary_columns):

        nseq = max(len(in_seq["itemId"]) - seq_length - 1, min(1, len(in_seq["itemId"])))

        item_id_seqs = [in_seq["itemId"][i : i + seq_length] for i in range(nseq)]
        targets = [in_seq["itemId"][i + seq_length] for i in range(nseq)]
        seqs = {"itemId": item_id_seqs}

        for sup_col in supplementary_columns:
            seqs[sup_col] = [in_seq[sup_col][i : i + seq_length] for i in range(nseq)]

        if len(seqs["itemId"]) == 1:
            seqs = {col: [[0] * (seq_length - len(seqs[col][0])) + seqs[col][0]] for col in ["itemId"] + supplementary_columns}

        return (seqs, targets)

    @classmethod
    def extract_sequences(cls, data, seq_length, supplementary_columns):

        raw_sequences = (
            data.sort_values(["sequenceId", "itemPosition"])
            .groupby("sequenceId")
            .agg({**{"itemId": list}, **{col: list for col in supplementary_columns}})
            .reset_index(drop=False)
        )
        col_types = {
            col: str(data[col].dtype) for col in ["itemId"] + supplementary_columns
        }
        rows = []
        for _, in_row in raw_sequences.iterrows():
            seqs, targets = cls.extract_subsequences(in_row[["itemId"]+supplementary_columns], seq_length, supplementary_columns)
            item_id_seqs = seqs.pop("itemId")
            for i, (seq, target) in enumerate(zip(item_id_seqs, targets)):
                subsequence_id = i
                rows.append([in_row["sequenceId"]] + [subsequence_id, "itemId"] + seq + [target])

                for sup_col, sup_col_seqs in seqs.items():
                    rows.append([in_row["sequenceId"]] + [subsequence_id, sup_col] + sup_col_seqs[i] + [None])
                
        sequences = pd.DataFrame(
            rows, columns=["sequenceId", "subsequenceId", "input_col"] + list(range(seq_length, 0, -1)) + ["target"]
        )
        return sequences, col_types

    @classmethod
    def get_subset_groups(cls, sequence_data, groups, n_cols):
        n_cases = int(sequence_data.shape[0]/n_cols)
        subset_groups = [([i]*math.floor(n_cases*size)) for i, size in enumerate(groups)]
        subset_groups = [inner for outer in subset_groups for inner in outer]
        diff = n_cases - len(subset_groups)
        subset_groups = ([0]*diff) + subset_groups
        return subset_groups

    @classmethod
    def extract_data_subsets(cls, sequences, groups):
        assert abs(1.0 - np.sum(groups)) < 0.0000000000001, np.sum(groups)

        datasets = [[] for _ in range(len(groups))]
        n_cols = len(np.unique(sequences["input_col"]))
        for _, sequence_data in sequences.groupby("sequenceId"):
            subset_groups = cls.get_subset_groups(sequence_data, groups, n_cols)
            assert len(subset_groups)*n_cols == sequence_data.shape[0]
            for i, group in enumerate(subset_groups):
                case_start = (i*n_cols)
                datasets[group].append(sequence_data.iloc[case_start:case_start+n_cols, :])

        return [pd.concat(dataset, axis=0) for dataset in datasets]


def preprocess(args, args_config):
    config = load_preprocessor_config(args.config_path, args_config)
    Preprocessor(**config.dict())
    print("Preprocessing complete")
