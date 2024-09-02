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

    group_proportions: List[float]
    seq_length: int
    seq_step_size: Optional[int]
    max_rows: Optional[int]
    seed: int
    n_cores: Optional[int]

    @validator("data_path", always=True)
    def validate_data_path(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"{v} does not exist")
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

    def __init__(self, **kwargs):
        kwargs["seq_step_size"] = kwargs.get("seq_step_size", kwargs["seq_length"])

        super().__init__(
            **{
                k: v
                for k, v in kwargs.items()
            }
        )
