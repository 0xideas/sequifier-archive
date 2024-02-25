import os
import shutil
import time
import yaml
import pandas as pd
import pytest



@pytest.fixture(scope="session")
def project_path():
    return os.path.join("tests", "project_folder")


@pytest.fixture(scope="session")
def preprocessing_config_paths():
    return [os.path.join("tests", "configs", file_name) for file_name in ["preprocess-test.yaml", "preprocess-test-3.yaml", "preprocess-test-5.yaml"]]


@pytest.fixture(scope="session")
def training_config_path():
    return os.path.join("tests", "configs", "train-test.yaml")


@pytest.fixture(scope="session")
def inference_config_path():
    return os.path.join("tests", "configs", "infer-test.yaml")


@pytest.fixture(scope="session")
def remove_project_path_contents(project_path):

    if os.path.exists(project_path):
        shutil.rmtree(project_path)
    os.makedirs(project_path)

    time.sleep(1)


def reformat_parameter(attr, param, type):
    if attr.endswith("_path"):
        if type == "linux->local":
            return(os.path.join(*param.split("/")))
        elif type == "local->linux":
            return("/".join(os.path.split(param)))
    else:
        return(param)



@pytest.fixture(scope="session", autouse=True)
def format_configs_locally(preprocessing_config_paths, training_config_path, inference_config_path):
    config_paths = preprocessing_config_paths + [training_config_path, inference_config_path]
    for config_path in config_paths:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config_formatted = {attr:reformat_parameter(attr, param, "linux->local") for attr, param in config.items()}
        
        with open(config_path, "w") as f:
            yaml.dump(config_formatted, f, default_flow_style=False, sort_keys=False)

    yield
    
    for config_path in config_paths:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config_formatted = {attr:reformat_parameter(attr, param, "local->linux") for attr, param in config.items()}
        
        with open(config_path, "w") as f:
            yaml.dump(config_formatted, f, default_flow_style=False, sort_keys=False)






@pytest.fixture(scope="session")
def run_preprocessing(preprocessing_config_paths, format_configs_locally, remove_project_path_contents):
    for preprocessing_config_path in preprocessing_config_paths:
        os.system(f"sequifier --preprocess --config_path={preprocessing_config_path}")


@pytest.fixture(scope="session")
def run_training(run_preprocessing, training_config_path):
    os.system(
        f"sequifier --train --on-preprocessed --config_path={training_config_path}"
    )


@pytest.fixture(scope="session")
def delete_inference_target(run_preprocessing, project_path, inference_config_path):
    with open(inference_config_path, "r") as f:
        config = yaml.safe_load(f)
    inference_data_path = os.path.join(project_path, config["inference_data_path"])

    inference_data = pd.read_csv(
        inference_data_path,
        sep=",",
        decimal=".",
        index_col=None,
    )

    inference_data = inference_data.drop(columns=["target"])

    inference_data.to_csv(inference_data_path)


@pytest.fixture(scope="session")
def run_inference(
    run_training, delete_inference_target, project_path, inference_config_path
):
    os.system(f"sequifier --infer --config_path={inference_config_path}")
