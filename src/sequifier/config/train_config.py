import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Union

import yaml
from pydantic import BaseModel, validator

ANYTYPE = Union[str, int, float]


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DotDict(copy.deepcopy(dict(self), memo=memo))


CustomValidation = Optional

VALID_LOSS_FUNCTIONS = [
    "L1Loss",
    "MSELoss",
    "CrossEntropyLoss",
    "CTCLoss",
    "NLLLoss",
    "PoissoLLoss",
    "GaussiaLLoss",
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


@dataclass
class TrainingSpecModel(BaseModel):
    device: str
    epochs: int
    iter_save: int
    batch_size: int
    lr: float  # learning rate
    dropout: float
    criterion: str
    optimizer: CustomValidation[DotDict]  # mandatory
    scheduler: CustomValidation[DotDict]  # mandatory
    continue_training: bool

    def __init__(self, **kwargs):

        super().__init__(
            **{k: v for k, v in kwargs.items() if k not in ["optimizer", "scheduler"]}
        )

        optimizer = kwargs.get("optimizer")
        scheduler = kwargs.get("scheduler")
        self.validate_optimizer_config(optimizer)
        self.optimizer = DotDict(optimizer)
        self.validate_scheduler_config(scheduler)
        self.scheduler = DotDict(scheduler)

    @validator("criterion")
    def validate_criterion(cls, v):
        if v not in VALID_LOSS_FUNCTIONS:
            raise ValueError(f"criterion must be in {VALID_LOSS_FUNCTIONS}")
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


class TransformerModel(BaseModel):
    project_path: str
    model_name: Optional[str]
    seq_length: int
    n_classes: int
    training_data_path: str
    validation_data_path: str
    seed: int

    model_spec: CustomValidation[ModelSpecModel]
    training_spec: CustomValidation[TrainingSpecModel]

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


def load_transformer_config(config_path, args_config, on_preprocessed):
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)

    config_values.update(args_config)

    if on_preprocessed:
        dd_config_path = os.path.join(
            config_values["project_path"], config_values.pop("ddconfig_path")
        )
        with open(dd_config_path, "r") as f:
            dd_config = json.loads(f.read())

        config_values["n_classes"] = dd_config["n_classes"]
        config_values["training_data_path"] = dd_config["split_paths"][0]
        config_values["validation_data_path"] = dd_config["split_paths"][1]

    return TransformerModel(**config_values)
