import os
from typing import List, Optional

import yaml
from pydantic import BaseModel, validator


def load_preprocessor_config(config_path, args_config):
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)

    config_values.update(args_config)

    return PreprocessorModel(**config_values)


class PreprocessorModel(BaseModel):
    project_path: str
    data_path: str
    read_format: str = "csv"
    write_format: str = "parquet"
    selected_columns: Optional[list[str]]
    target_column: Optional[str]
    return_targets: bool = True

    group_proportions: List[float]
    seq_length: int
    max_rows: Optional[int]
    seed: int
    n_cores: Optional[int]

    @validator("data_path", always=True)
    def validate_data_path(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"{v} does not exist")
        return v

    @validator("group_proportions", always=True)
    def validate_group_proportions(cls, v):
        if abs(sum(v) - 1) > 1e-10:
            raise ValueError(f"does not sum to 1: {v} - {sum(v)}")
        if len(v) < 3:
            raise ValueError(
                f"You need at least 3 splits of data, which correspond to training, validation and testing data"
            )
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

    @validator("return_targets", always=True)
    def validate_return_targets(cls, v, values):
        assert (
            v is False or values["target_column"] is not None
        ), "Either return_targets is False or target_column is not None"
        assert not (
            values["target_column"] is not None and v is False
        ), "If return_targets is False, target_column has to be None"
        return v
