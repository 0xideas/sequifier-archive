import os
from typing import Optional

import yaml
from pydantic import BaseModel, validator


class InfererModel(BaseModel):
    project_path: str
    inference_data_path: str
    model_path: str
    device: str
    seq_length: int
    ddconfig_path: Optional[str] = None
    output_probabilities: bool = False
    map_to_id: bool = True
    seed: int

    @validator("inference_data_path")
    def validate_inference_data_path(cls, v, values):
        path = os.path.join(values["project_path"], v)

        if not os.path.exists(path):
            raise ValueError(f"{path} does not exist")

        return v


def load_inferer_config(config_path, args_config):
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)
    config_values.update(args_config)

    return InfererModel(**config_values)
