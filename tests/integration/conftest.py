import os
import shutil
import time

import pytest


@pytest.fixture(scope="session")
def project_path():
    return os.path.join("tests", "project_folder")


@pytest.fixture(scope="session")
def preprocessing_config_path():
    return os.path.join("tests", "configs", "preprocess", "test.yaml")


@pytest.fixture(scope="session")
def training_config_path():
    return os.path.join("tests", "configs", "train", "test.yaml")


@pytest.fixture(scope="session")
def inference_config_path():
    return os.path.join("tests", "configs", "infer", "test.yaml")


@pytest.fixture(scope="session")
def remove_project_path_contents(project_path):

    shutil.rmtree(project_path)
    os.makedirs(project_path)

    time.sleep(1)


@pytest.fixture(scope="session")
def run_preprocessing(
    project_path, preprocessing_config_path, remove_project_path_contents
):
    os.system(
        f"sequifier --preprocess --config_path={preprocessing_config_path} --project_path={project_path}"
    )


@pytest.fixture(scope="session")
def run_training(run_preprocessing, project_path, training_config_path):
    os.system(
        f"sequifier --train --on-preprocessed --config_path={training_config_path} --project_path={project_path}"
    )


@pytest.fixture(scope="session")
def run_inference(run_training, project_path, inference_config_path):
    os.system(
        f"sequifier --infer --config_path={inference_config_path} --project_path={project_path}"
    )
