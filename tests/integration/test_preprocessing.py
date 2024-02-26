import json
import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def dd_configs(run_preprocessing, project_path):
    dd_configs = {}
    for data_number in [1, 3, 5]:
        for variant in ["", "_real"]:
            file_name = f"test_data{variant}_{data_number}.json"
            with open(
                os.path.join(project_path, "configs", "ddconfigs", file_name), "r"
            ) as f:
                dd_conf = json.loads(f.read())
            dd_configs[file_name] = dd_conf
    return dd_configs


def test_dd_config(dd_configs):
    for file_name, dd_config in dd_configs.items():
        print(file_name)
        assert np.all(
            np.array(list(dd_config.keys()))
            == np.array(["n_classes", "id_maps", "split_paths", "column_types"])
        ), list(dd_config.keys())

        assert len(dd_config["split_paths"]) == 3
        assert dd_config["split_paths"][0].endswith("split0.csv")

        if "itemId" in dd_config["n_classes"]:
            assert len(dd_config["id_maps"]["itemId"]) == 30
            assert dd_config["n_classes"]["itemId"] == 31

            id_map_keys = np.array(sorted(list(dd_config["id_maps"]["itemId"].keys())))
            # assert False, np.array([str(x) for x in range(100, 130)])
            assert np.all(id_map_keys == np.array([str(x) for x in range(100, 130)]))

        for col in dd_config["id_maps"].keys():
            id_map_values = np.array(sorted(list(dd_config["id_maps"][col].values())))
            # assert False, id_map_values
            assert np.all(
                id_map_values == np.arange(1, len(id_map_values) + 1)
            ), id_map_values


@pytest.fixture()
def data_splits(project_path):
    data_split_values = {
        f"{j}{variant}": [
            pd.read_csv(
                os.path.join(
                    project_path, "data", f"test_data{variant}_1-split{i}.csv"
                ),
                sep=",",
                decimal=".",
                index_col=None,
            )
            for i in range(3)
        ]
        for variant in ["", "_real"]
        for j in [1, 3, 5]
    }

    return data_split_values


def test_preprocessed_data(data_splits):
    for variant in ["", "_real"]:
        for j in [1, 3, 5]:
            name = f"{j}{variant}"
            assert len(data_splits[name]) == 3

            for data in data_splits[name]:
                assert data.shape[1] == 12
                sequence_step = (
                    data["sequenceId"].values[:-1] != data["sequenceId"].values[1:]
                )
                assert np.all(
                    (data["1"].values[:-1] == data["2"].values[1:]) | sequence_step
                )
                assert np.all(
                    (data["7"].values[:-1] == data["8"].values[1:]) | sequence_step
                )
