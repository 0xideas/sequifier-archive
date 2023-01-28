import json
import math
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

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

        n_classes = len(np.unique(data["itemId"])) + 1

        data, id_map = self.replace_ids(data)

        sequences = self.extract_sequences(data, seq_length)

        self.splits = self.extract_data_subsets(sequences, group_proportions)
        self.splits = [self.cast_columns_to_string(data) for data in self.splits]
        self.export(id_map, n_classes)

    def export(self, id_map, n_classes):

        data_driven_config = {
            "n_classes": n_classes,
            "id_map": id_map,
            "split_paths": self.split_paths,
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
    def replace_ids(cls, data):
        ids = sorted(
            [int(x) if not isinstance(x, str) else x for x in np.unique(data["itemId"])]
        )
        id_map = {id_: i + 1 for i, id_ in enumerate(ids)}
        data["itemId"] = data["itemId"].map(id_map)
        return (data, id_map)

    @classmethod
    def extract_subsequences(cls, in_seq, seq_length):
        nseq = np.max([len(in_seq) - seq_length - 1, np.min([1, len(in_seq)])])

        seqs = [in_seq[i : i + seq_length] for i in range(nseq)]
        targets = [in_seq[i + seq_length] for i in range(nseq)]

        if len(seqs) == 1:
            seqs = [[0] * (seq_length - len(seqs[0])) + seqs[0]]

        return (seqs, targets)

    @classmethod
    def extract_sequences(cls, data, seq_length):
        raw_sequences = (
            data.sort_values(["sequenceId", "timesort"])
            .groupby("sequenceId")["itemId"]
            .apply(list)
            .reset_index(drop=False)
        )
        rows = []
        for _, in_row in raw_sequences.iterrows():
            seqs, targets = cls.extract_subsequences(in_row["itemId"], seq_length)
            for seq, target in zip(seqs, targets):
                rows.append([in_row["sequenceId"]] + seq + [target])
        sequences = pd.DataFrame(
            rows, columns=["sequenceId"] + list(range(seq_length, 0, -1)) + ["target"]
        )
        return sequences

    @classmethod
    def get_subset_indices(cls, user_data, groups):
        subset_indices = [math.floor(size * user_data.shape[0]) for size in groups]
        diff = user_data.shape[0] - np.sum(subset_indices)

        additional = np.random.choice(range(len(groups)), replace=True, size=diff)
        for i in additional:
            subset_indices[i] += 1

        return subset_indices

    @classmethod
    def extract_data_subsets(cls, sequences, groups):
        assert abs(1 - np.sum(groups)) < 0.99999999999, np.sum(groups)

        datasets = [[] for _ in range(len(groups))]
        for _, user_data in sequences.groupby("sequenceId"):
            subset_indices = cls.get_subset_indices(user_data, groups)
            indices = list(np.cumsum(subset_indices))
            for i, (start, end) in enumerate(zip([0] + indices[:-1], indices)):
                datasets[i].append(user_data.iloc[start:end, :])

        return [pd.concat(dataset, axis=0) for dataset in datasets]


def preprocess(args, args_config):
    config = load_preprocessor_config(args.config_path, args_config)
    Preprocessor(**config.dict())
    print("Preprocessing complete")
