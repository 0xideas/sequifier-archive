import os
import shutil
import time

import pandas as pd
import pytest
import yaml

SELECTED_COLUMNS = {
    "categorical": {
        1: "itemId",
        3: "itemId,sup1",
        5: "itemId,sup1,sup2,sup4",
    },
    "real": {
        1: "itemValue",
        3: "itemValue,sup1,sup2",
        5: "itemValue,sup1,sup2,sup3,sup4",
    },
}


@pytest.fixture(scope="session")
def project_path():
    return os.path.join("tests", "project_folder")


@pytest.fixture(scope="session")
def preprocessing_config_path_cat():
    return os.path.join("tests", "configs", "preprocess-test-categorical.yaml")


@pytest.fixture(scope="session")
def preprocessing_config_path_real():
    return os.path.join("tests", "configs", "preprocess-test-real.yaml")


@pytest.fixture(scope="session")
def training_config_path_cat():
    return os.path.join("tests", "configs", "train-test-categorical.yaml")


@pytest.fixture(scope="session")
def training_config_path_real():
    return os.path.join("tests", "configs", "train-test-real.yaml")


@pytest.fixture(scope="session")
def inference_config_path_cat():
    return os.path.join("tests", "configs", "infer-test-categorical.yaml")


@pytest.fixture(scope="session")
def inference_config_path_real():
    return os.path.join("tests", "configs", "infer-test-real.yaml")


@pytest.fixture(scope="session")
def inference_config_path_real_autoregression():
    return os.path.join("tests", "configs", "infer-test-real-autoregression.yaml")


@pytest.fixture(scope="session")
def remove_project_path_contents(project_path):

    if os.path.exists(project_path):
        shutil.rmtree(project_path)
    os.makedirs(project_path)

    time.sleep(1)


def reformat_parameter(attr, param, type):
    if attr.endswith("_path"):
        if type == "linux->local":
            return os.path.join(*param.split("/"))
        elif type == "local->linux":
            return "/".join(os.path.split(param))
    else:
        return param


@pytest.fixture(scope="session", autouse=True)
def format_configs_locally(
    preprocessing_config_path_cat,
    preprocessing_config_path_real,
    training_config_path_cat,
    training_config_path_real,
    inference_config_path_cat,
    inference_config_path_real,
    inference_config_path_real_autoregression,
):
    config_paths = [
        preprocessing_config_path_cat,
        preprocessing_config_path_real,
        training_config_path_cat,
        training_config_path_real,
        inference_config_path_cat,
        inference_config_path_real,
        inference_config_path_real_autoregression,
    ]
    for config_path in config_paths:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        assert config is not None, config_path

        config_formatted = {
            attr: reformat_parameter(attr, param, "linux->local")
            for attr, param in config.items()
        }

        with open(config_path, "w") as f:
            yaml.dump(config_formatted, f, default_flow_style=False, sort_keys=False)

    yield

    for config_path in config_paths:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config_formatted = {
            attr: reformat_parameter(attr, param, "local->linux")
            for attr, param in config.items()
        }

        with open(config_path, "w") as f:
            yaml.dump(config_formatted, f, default_flow_style=False, sort_keys=False)


@pytest.fixture(scope="session")
def run_preprocessing(
    preprocessing_config_path_cat,
    preprocessing_config_path_real,
    format_configs_locally,
    remove_project_path_contents,
):
    for data_number in [1, 3, 5]:
        data_path_cat = os.path.join(
            "tests", "resources", f"test_data_categorical_{data_number}.csv"
        )
        os.system(
            f"sequifier --preprocess --config-path={preprocessing_config_path_cat} --data-path={data_path_cat} --selected-columns=None"
        )

        data_path_real = os.path.join(
            "tests", "resources", f"test_data_real_{data_number}.csv"
        )
        os.system(
            f"sequifier --preprocess --config-path={preprocessing_config_path_real} --data-path={data_path_real} --selected-columns={SELECTED_COLUMNS['real'][data_number]}"
        )

    source_path = os.path.join(
        "tests", "resources", "test_data_real_1-split2-autoregression.csv"
    )

    target_path = os.path.join(
        "tests", "project_folder", "data", "test_data_real_1-split2-autoregression.csv"
    )

    shutil.copyfile(source_path, target_path)


@pytest.fixture(scope="session")
def delete_inference_target(
    run_preprocessing,
    project_path,
):

    inference_data_paths = [
        os.path.join(
            project_path, "data", f"test_data_{variant}_{model_number}-split2.csv"
        )
        for variant in ["categorical", "real"]
        for model_number in [1, 3, 5]
    ]

    inference_data_paths.append(
        os.path.join(
            project_path, "data", f"test_data_real_1-split2-autoregression.csv"
        )
    )

    for inference_data_path in inference_data_paths:

        inference_data = pd.read_csv(
            inference_data_path,
            sep=",",
            decimal=".",
            index_col=None,
        )

        inference_data = inference_data.drop(columns=["target"])

        inference_data.to_csv(inference_data_path, index=None, decimal=".", sep=",")


@pytest.fixture(scope="session")
def run_training(
    run_preprocessing, project_path, training_config_path_cat, training_config_path_real
):
    for model_number in [1, 3, 5]:
        ddconfig_path_cat = os.path.join(
            "configs", "ddconfigs", f"test_data_categorical_{model_number}.json"
        )
        model_name_cat = f"model-categorical-{model_number}"
        os.system(
            f"sequifier --train --on-preprocessed --config-path={training_config_path_cat} --ddconfig-path={ddconfig_path_cat} --model-name={model_name_cat} --selected-columns={SELECTED_COLUMNS['categorical'][model_number]}"
        )

        ddconfig_path_real = os.path.join(
            "configs", "ddconfigs", f"test_data_real_{model_number}.json"
        )
        model_name_real = f"model-real-{model_number}"
        os.system(
            f"sequifier --train --on-preprocessed --config-path={training_config_path_real} --ddconfig-path={ddconfig_path_real} --model-name={model_name_real} --selected-columns=None"
        )

    source_path = os.path.join(
        project_path, "models", "sequifier-model-real-1-best-3.onnx"
    )
    target_path = os.path.join(
        project_path, "models", "sequifier-model-real-1-best-3-autoregression.onnx"
    )

    shutil.copy(source_path, target_path)


@pytest.fixture(scope="session")
def run_inference(
    run_training,
    delete_inference_target,
    project_path,
    inference_config_path_cat,
    inference_config_path_real,
    inference_config_path_real_autoregression,
):
    for model_number in [1, 3, 5]:
        inference_model_path_cat = os.path.join(
            "models", f"sequifier-model-categorical-{model_number}-best-3.onnx"
        )
        inference_data_path_cat = os.path.join(
            "data", f"test_data_categorical_{model_number}-split2.csv"
        )
        ddconfig_path_cat = os.path.join(
            "configs", "ddconfigs", f"test_data_categorical_{model_number}.json"
        )
        os.system(
            f"sequifier --infer --on-preprocessed --config-path={inference_config_path_cat} --ddconfig-path={ddconfig_path_cat} --inference-model-path={inference_model_path_cat} --inference-data-path={inference_data_path_cat} --selected-columns={SELECTED_COLUMNS['categorical'][model_number]}"
        )

        inference_model_path_real = os.path.join(
            "models", f"sequifier-model-real-{model_number}-best-3.onnx"
        )
        inference_data_path_real = os.path.join(
            "data", f"test_data_real_{model_number}-split2.csv"
        )
        ddconfig_path_real = os.path.join(
            "configs", "ddconfigs", f"test_data_real_{model_number}.json"
        )
        os.system(
            f"sequifier --infer --on-preprocessed --config-path={inference_config_path_real} --ddconfig-path={ddconfig_path_real} --inference-model-path={inference_model_path_real} --inference-data-path={inference_data_path_real} --selected-columns=None"
        )

    os.system(
        f"sequifier --infer --on-preprocessed --config-path={inference_config_path_real_autoregression} --selected-columns={SELECTED_COLUMNS['real'][1]}"
    )
