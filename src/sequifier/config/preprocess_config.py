import yaml
import os
from pydantic import BaseModel, validator
from typing import List, Optional

class PreprocessorModel(BaseModel):
    project_path: str
    data_path: str
    seq_length: int
    group_proportions: List[float]
    max_rows: Optional[int]
    seed: int


    @validator("data_path")
    def validate_data_path(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"{v} does not exist")
        return(v)

    @validator("group_proportions")
    def validate_group_proportions(cls, v):
        if abs(sum(v)- 1) > 1e-10:
            raise ValueError(f"does not sum to 1: {v} - {sum(v)}")
        if len(v) < 3:
            raise ValueError(f"You need at least 3 splits of data, which correspond to training, validation and testing data")
        return(v)


def load_preprocessor_config(config_path, args_config):
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)

    config_values.update(args_config)
    

    return(PreprocessorModel(**config_values))