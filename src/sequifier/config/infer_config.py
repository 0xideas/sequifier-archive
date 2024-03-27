import json
import os
from typing import Optional

import yaml
from pydantic import BaseModel, validator


class InfererModel(BaseModel):
    project_path: str
    inference_data_path: str
    inference_model_path: str
    device: str
    seq_length: int
    ddconfig_path: Optional[str] = None
    output_probabilities: bool = False
    map_to_id: bool = True
    seed: int
    selected_columns: Optional[list[str]]
    column_types: dict[str, str]
    categorical_columns: list[str]
    real_columns: list[str]
    target_column: str
    target_column_type: str
    sample_from_distribution: bool = False
    infer_with_dropout: bool = False
    sample_from_distribution: Optional[bool] = False
    inference_batch_size: int
    auto_regression: bool = True
    read_format: str = "parquet"
    write_format: str = "csv"
    training_config_path: Optional[str] = None

    @validator("inference_data_path", always=True)
    def validate_inference_data_path(cls, v, values):
        path = os.path.join(values["project_path"], v)

        if not os.path.exists(path):
            raise ValueError(f"{path} does not exist")

        return v

    @validator("sample_from_distribution", always=True)
    def validate_sample_from_distribution(cls, v, values):

        if v and values["target_column_type"] == "real":
            raise ValueError(
                "sample_from_distribution can only be set to true for categorical target variables"
            )

        return v

    @validator("target_column_type", always=True)
    def validate_target_column_type(cls, v):
        assert v in ["categorical", "real"], v
        return v

    @validator("read_format", always=True)
    def validate_read_format(cls, v):
        assert v in [
            "csv",
            "parquet",
        ], "Currently only 'csv' and 'parquet' are supported"
        return v

    @validator("training_config_path", always=True)
    def validate_training_config_path(cls, v, values):
        assert v is not None or values["inference_model_path"].endswith(".onnx")
        return v

    @validator("write_format", always=True)
    def validate_write_format(cls, v):
        assert v in [
            "csv",
            "parquet",
        ], "Currently only 'csv' and 'parquet' are supported"
        return v

    def __init__(self, **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items()})
        if self.target_column_type == "real":
            assert not self.output_probabilities


def load_inferer_config(config_path, args_config, on_unprocessed):
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)
    config_values.update(args_config)

    if not on_unprocessed:
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
