import json
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
    column_types: dict[str, str]
    categorical_columns: list[str]
    real_columns: list[str]
    target_column: str
    target_column_type: str
    batch_size: int

    @validator("inference_data_path")
    def validate_inference_data_path(cls, v, values):
        path = os.path.join(values["project_path"], v)

        if not os.path.exists(path):
            raise ValueError(f"{path} does not exist")

        return v

    def __init__(self, **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items()})
        assert self.target_column_type in ["categorical", "real"]
        if self.target_column_type == "real":
            assert not self.output_probabilities


def load_inferer_config(config_path, args_config, on_preprocessed):
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)
    config_values.update(args_config)

    if on_preprocessed:
        dd_config_path = os.path.join(
            config_values["project_path"], config_values.get("ddconfig_path")
        )

        with open(dd_config_path, "r") as f:
            dd_config = json.loads(f.read())

        config_values["column_types"] = dd_config["column_types"]
        config_values["categorical_columns"] = [
            col for col, type_ in dd_config["column_types"].items() if type_ == "int64"
        ]
        config_values["real_columns"] = [
            col
            for col, type_ in dd_config["column_types"].items()
            if type_ == "float64"
        ]
    return InfererModel(**config_values)
