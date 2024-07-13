import json
import os
from typing import Optional

import numpy as np
import yaml
from pydantic import BaseModel, validator

from sequifier.helpers import normalize_path


def load_inferer_config(config_path, args_config, on_unprocessed):
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)
    config_values.update(args_config)

    if not on_unprocessed:
        dd_config_path = config_values.get("ddconfig_path")

        with open(
            normalize_path(dd_config_path, config_values["project_path"]), "r"
        ) as f:
            dd_config = json.loads(f.read())

        config_values["column_types"] = config_values.get(
            "column_types", dd_config["column_types"]
        )

        if config_values["selected_columns"] is None:
            config_values["selected_columns"] = list(
                config_values["column_types"].keys()
            )

        config_values["categorical_columns"] = [
            col
            for col, type_ in dd_config["column_types"].items()
            if type_ == "int64" and col in config_values["selected_columns"]
        ]
        config_values["real_columns"] = [
            col
            for col, type_ in dd_config["column_types"].items()
            if type_ == "float64" and col in config_values["selected_columns"]
        ]
        config_values["data_path"] = normalize_path(
            config_values.get(
                "data_path",
                dd_config["split_paths"][min(2, len(dd_config["split_paths"]) - 1)],
            ),
            config_values["project_path"],
        )

    return InfererModel(**config_values)


class InfererModel(BaseModel):
    project_path: str
    ddconfig_path: Optional[str] = None
    model_path: str
    data_path: str
    training_config_path: Optional[str] = None
    read_format: str = "parquet"
    write_format: str = "csv"

    selected_columns: Optional[list[str]]
    categorical_columns: list[str]
    real_columns: list[str]
    target_columns: list[str]
    column_types: dict[str, str]
    target_column_types: dict[str, str]

    output_probabilities: bool = False
    map_to_id: bool = True
    seed: int
    device: str
    seq_length: int
    inference_batch_size: int

    sample_from_distribution: bool = False
    infer_with_dropout: bool = False
    autoregression: bool = True
    autoregression_additional_steps: Optional[int] = None

    @validator("training_config_path", always=True)
    def validate_training_config_path(cls, v):
        if not (v is None or os.path.exists(v)):
            raise ValueError(f"{v} does not exist")

        return v

    @validator("data_path", always=True)
    def validate_data_path(cls, v, values):
        v2 = normalize_path(v, values["project_path"])
        if not os.path.exists(v2):
            raise ValueError(f"{v2} does not exist")

        return v

    @validator("read_format", always=True)
    def validate_read_format(cls, v):
        assert v in [
            "csv",
            "parquet",
        ], "Currently only 'csv' and 'parquet' are supported"
        return v

    @validator("write_format", always=True)
    def validate_write_format(cls, v):
        assert v in [
            "csv",
            "parquet",
        ], "Currently only 'csv' and 'parquet' are supported"
        return v

    @validator("target_column_types", always=True)
    def validate_target_column_types(cls, v, values):
        assert np.all([vv in ["categorical", "real"] for vv in v.values()])
        assert np.all(
            np.array(list(v.keys())) == np.array(values["target_columns"])
        ), "target_columns and target_column_types must contain the same values/keys in the same order"
        return v

    @validator("map_to_id", always=True)
    def validate_map_to_id(cls, v, values):
        assert v == False or np.max(
            np.array(list(values["target_column_types"].values())) == "categorical"
        ), f"map_to_id can only be True if at least one target variable is categorical: {np.array(values['target_column_types'].values()) == 'categorical'}"
        return v

    @validator("sample_from_distribution", always=True)
    def validate_sample_from_distribution(cls, v, values):
        if (
            v
            and np.max(np.array(list(values["target_column_types"].values())) == "real")
            == 1
        ):
            raise ValueError(
                "sample_from_distribution can only be used when all target columns are categorical"
            )

        return v

    def __init__(self, **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items()})

        column_ordered = np.array(list(self.column_types.keys()))
        columns_ordered_filtered = column_ordered[
            np.array([c in self.target_columns for c in column_ordered])
        ]
        assert np.all(
            columns_ordered_filtered == np.array(self.target_columns)
        ), f"{columns_ordered_filtered = } != {self.target_columns = }"
