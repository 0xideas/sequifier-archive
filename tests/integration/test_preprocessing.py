import os
import json
import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def config_path():
    return ("tests/configs/preprocess/test.yaml")


@pytest.fixture()
def run_preprocessing(project_path, config_path):
    os.system(f"sequifier --preprocess --config_path={config_path} --project_path={project_path}")  


@pytest.fixture()
def dd_config(run_preprocessing, project_path):
    with open(f"{project_path}/configs/ddconfigs/test_data.json", "r") as f:
        dd_conf = json.loads(f.read())
    return(dd_conf)


def test_dd_config(dd_config):
    assert np.all(np.array(list(dd_config.keys()))==np.array(["n_classes", "id_map", "split_paths"]))
    assert dd_config["n_classes"] == 31
    assert len(dd_config["split_paths"]) == 3
    assert dd_config["split_paths"][0].endswith("test_data-split0.csv")
    assert len(dd_config["id_map"]) == 30

    id_map_keys = np.array(sorted(list(dd_config["id_map"].keys())))
    # assert False, np.array([str(x) for x in range(100, 130)])
    assert np.all(id_map_keys == np.array([str(x) for x in range(100, 130)]))

    id_map_values = np.array(sorted(list(dd_config["id_map"].values())))
    # assert False, id_map_values
    assert np.all(id_map_values == np.arange(1, 31)), id_map_values


@pytest.fixture()
def data_splits(project_path):
    data_splits_paths = [f"{project_path}/data/test_data-split{i}.csv" for i in range(3)]

    data_split_values = [pd.read_csv(path, sep=",", decimal=".", index_col=None) for path in data_splits_paths]

    return(data_split_values)


def test_preprocessed_data(data_splits):
    assert np.all(np.array([data.shape[0] for data in data_splits])==np.array([51, 18, 21]))

    for data in data_splits:
        assert data.shape[1] == 12
        sequence_step = data["sequenceId"].values[:-1] != data["sequenceId"].values[1:]
        assert np.all((data["target"].values[:-1] == data["1"].values[1:]) | sequence_step)
        assert np.all((data["1"].values[:-1] == data["2"].values[1:]) | sequence_step)
        assert np.all((data["7"].values[:-1] == data["8"].values[1:]) | sequence_step)
