import json
import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def dd_configs(run_preprocessing, project_path):
    dd_configs = {}
    for data_number in [1, 3, 5]:
        for variant in ["categorical", "real"]:
            file_name = f"test_data_{variant}_{data_number}.json"
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
        assert dd_config["split_paths"][0].endswith("split0.parquet")

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
        f"{j}_{variant}": [
            pd.read_parquet(
                os.path.join(
                    project_path, "data", f"test_data_{variant}_{j}-split{i}.parquet"
                )
            )
            for i in range(3)
        ]
        for variant in ["categorical", "real"]
        for j in [1, 3, 5]
    }

    return data_split_values


def test_preprocessed_data_real(delete_inference_target, data_splits):
    for j in [1, 3, 5]:
        name = f"{j}_real"
        assert len(data_splits[name]) == 3

        for i, data in enumerate(data_splits[name]):
            number_expected_columns = 12 - int(i == 2)
            assert data.shape[1] == (
                number_expected_columns
            ), f"{name = } - {i = }: {data.shape = } - {data.columns = }"
            for sequenceId, group in data.groupby("sequenceId"):

                # offset by j in either direction as that is the number of columns in the input
                # data, thus an offset by 1 'observation' requires an offset by j values
                assert np.all((group["1"].values[:-j] == group["2"].values[j:]))
                assert np.all((group["5"].values[:-j] == group["6"].values[j:]))


def test_preprocessed_data_categorical(data_splits):
    for j in [1, 3, 5]:
        name = f"{j}_categorical"
        assert len(data_splits[name]) == 3

        for i, data in enumerate(data_splits[name]):
            number_expected_columns = 12 - int(i == 2)
            assert data.shape[1] == (
                number_expected_columns
            ), f"{name = } - {i = }: {data.shape = } - {data.columns = }"

            for sequenceId, group in data.groupby("sequenceId"):

                # offset by j in either direction as that is the number of columns in the input
                # data, thus an offset by 1 'observation' requires an offset by j values
                assert np.all(
                    np.abs(group["1"].values[:-j] - group["2"].values[j:]) < 0.0001
                ), f'{list(group["1"].values[:-j]) = } != {list(group["2"].values[j:]) = }'
                assert np.all(
                    np.abs(group["5"].values[:-j] - group["6"].values[j:]) < 0.0001
                ), f'{list(group["5"].values[:-j]) = } != {list(group["6"].values[j:]) = }'
