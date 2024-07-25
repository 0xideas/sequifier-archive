import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import yaml
from pydantic import BaseModel, validator

from sequifier.helpers import normalize_path

ANYTYPE = Union[str, int, float]


def load_train_config(config_path, args_config, on_unprocessed):
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)

    config_values.update(args_config)

    if not on_unprocessed:
        dd_config_path = config_values.pop("ddconfig_path")

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
        config_values["n_classes"] = config_values.get(
            "n_classes", dd_config["n_classes"]
        )
        config_values["training_data_path"] = normalize_path(
            config_values.get("training_data_path", dd_config["split_paths"][0]),
            config_values["project_path"],
        )
        config_values["validation_data_path"] = normalize_path(
            config_values.get(
                "validation_data_path",
                dd_config["split_paths"][min(1, len(dd_config["split_paths"]) - 1)],
            ),
            config_values["project_path"],
        )

        config_values["id_maps"] = dd_config["id_maps"]

    return TrainModel(**config_values)


VALID_LOSS_FUNCTIONS = [
    "L1Loss",
    "MSELoss",
    "CrossEntropyLoss",
    "CTCLoss",
    "NLLLoss",
    "PoissonNLLLoss",
    "GaussianNLLLoss",
    "KLDivLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "MarginRankingLoss",
    "HingeEmbeddingLoss",
    "MultiLabelMarginLoss",
    "HuberLoss",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "MultiLabelSoftMarginLoss",
    "CosineEmbeddingLoss",
    "MultiMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
]
VALID_OPTIMIZERS = [
    "Adadelta",
    "Adagrad",
    "Adam",
    "AdamW",
    "SparseAdam",
    "Adamax",
    "ASGD",
    "LBFGS",
    "NAdam",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
]
VALID_SCHEDULERS = [
    "LambdaLR",
    "MultiplicativeLR",
    "StepLR",
    "MultiStepLR",
    "ConstantLR",
    "LinearLR",
    "ExponentialLR",
    "PolynomialLR",
    "CosineAnnealingLR",
    "ChainedScheduler",
    "SequentialLR",
    "ReduceLROnPlateau",
    "CyclicLR",
    "OneCycleLR",
    "CosineAnnealingWarmRestarts",
]


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DotDict(copy.deepcopy(dict(self), memo=memo))


CustomValidation = Optional


@dataclass
class TrainingSpecModel(BaseModel):
    device: str
    epochs: int
    log_interval: int = 10
    class_share_log_columns: list[str] = []
    early_stopping_epochs: Optional[int]
    iter_save: int
    batch_size: int
    lr: float  # learning rate
    criterion: dict[str, str]
    accumulation_steps: Optional[int]
    dropout: float = 0.0
    loss_weights: Optional[dict[str, float]]
    optimizer: CustomValidation[DotDict]  # mandatory; default value in __init__
    scheduler: CustomValidation[DotDict]  # mandatory; default value in __init__
    continue_training: bool = True

    def __init__(self, **kwargs):
        super().__init__(
            **{k: v for k, v in kwargs.items() if k not in ["optimizer", "scheduler"]}
        )

        optimizer = kwargs.get("optimizer", {"name": "Adam"})
        scheduler = kwargs.get(
            "scheduler", {"name": "StepLR", "step_size": 1, "gamma": 0.99}
        )

        self.validate_optimizer_config(optimizer)
        self.optimizer = DotDict(optimizer)
        self.validate_scheduler_config(scheduler)
        self.scheduler = DotDict(scheduler)

    @validator("criterion")
    def validate_criterion(cls, v):
        for vv in v.values():
            if vv not in VALID_LOSS_FUNCTIONS:
                raise ValueError(
                    f"criterion must be in {VALID_LOSS_FUNCTIONS}, {vv} isn't"
                )
        return v

    @staticmethod
    def validate_optimizer_config(v):
        if "name" not in v:
            raise ValueError("optimizer dict must specify 'name' field")
        if v["name"] not in VALID_OPTIMIZERS:
            raise ValueError(f"optimizer not valid as not found in {VALID_OPTIMIZERS}")

    @staticmethod
    def validate_scheduler_config(v):
        if "name" not in v:
            raise ValueError("scheduler dict must specify 'name' field")
        if v["name"] not in VALID_SCHEDULERS:
            raise ValueError(f"scheduler not valid as not found in {VALID_SCHEDULERS}")


class ModelSpecModel(BaseModel):
    d_model: int
    nhead: int
    d_hid: int
    nlayers: int


class TrainModel(BaseModel):
    project_path: str
    model_name: Optional[str]
    training_data_path: str
    validation_data_path: str
    read_format: str = "parquet"

    selected_columns: Optional[list[str]]
    column_types: dict[str, str]
    categorical_columns: list[str]
    real_columns: list[str]
    target_columns: list[str]
    target_column_types: dict[str, str]
    id_maps: dict[str, dict[Union[str, int], int]]

    seq_length: int
    n_classes: dict[str, int]
    inference_batch_size: int
    seed: int

    export_onnx: bool = True
    export_pt: bool = False
    export_with_dropout: bool = False

    model_spec: CustomValidation[ModelSpecModel]
    training_spec: CustomValidation[TrainingSpecModel]

    @validator("target_column_types", always=True)
    def validate_target_column_types(cls, v, values):
        assert np.all([vv in ["categorical", "real"] for vv in v.values()])
        assert np.all(
            np.array(list(v.keys())) == np.array(values["target_columns"])
        ), "target_columns and target_column_types must contain the same values/keys in the same order"
        return v

    @validator("read_format", always=True)
    def validate_read_format(cls, v):
        assert v in [
            "csv",
            "parquet",
        ], "Currently only 'csv' and 'parquet' are supported"
        return v

    def __init__(self, **kwargs):
        super().__init__(
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["model_spec", "training_spec"]
            }
        )
        self.model_spec = ModelSpecModel(**kwargs.get("model_spec"))
        self.training_spec = TrainingSpecModel(**kwargs.get("training_spec"))

        assert np.all(
            np.array(self.target_columns)
            == np.array(list(self.training_spec.criterion.keys()))
        ), "target_columns and criterion must contain the same values/keys in the same order"

        column_ordered = np.array(list(self.column_types.keys()))
        columns_ordered_filtered = column_ordered[
            np.array([c in self.target_columns for c in column_ordered])
        ]
        assert np.all(
            columns_ordered_filtered == np.array(self.target_columns)
        ), f"{columns_ordered_filtered = } != {self.target_columns = }"
