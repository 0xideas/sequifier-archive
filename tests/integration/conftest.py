import os
import pytest




@pytest.fixture(scope="session")
def project_path():
    return ("tests/project_folder") 


@pytest.fixture(scope="session")
def preprocessing_config_path():
    return ("tests/configs/preprocess/test.yaml")


@pytest.fixture(scope="session")
def training_config_path():
    return ("tests/configs/train/test.yaml")


@pytest.fixture(scope="session")
def inference_config_path():
    return ("tests/configs/infer/test.yaml")


@pytest.fixture(scope="session")
def run_preprocessing(project_path, preprocessing_config_path):
    os.system(f"sequifier --preprocess --config_path={preprocessing_config_path} --project_path={project_path}")  


@pytest.fixture(scope="session")
def remove_old_checkpoints(project_path):
    checkpoint_path = f"{project_path}/checkpoints"
    for file in os.listdir(checkpoint_path):
        os.remove(os.path.join(checkpoint_path, file))


@pytest.fixture(scope="session")
def run_training(run_preprocessing, remove_old_checkpoints, project_path, training_config_path):
    os.system(f"sequifier --train --on-preprocessed --config_path={training_config_path} --project_path={project_path}")  


@pytest.fixture(scope="session")
def run_inference(run_training, project_path, inference_config_path):
    os.system(f"sequifier --infer --config_path={inference_config_path} --project_path={project_path}")  
