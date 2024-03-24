import json
import os

import numpy as np
import pandas as pd
import pytest


def test_checkpoint_files_exists(run_training, project_path):

    found_items = np.array(
        sorted(list(os.listdir(os.path.join(project_path, "checkpoints"))))
    )
    expected_items = np.array(
        [
            f"model-{model_type}-{j}-epoch-{i}.pt"
            for model_type in ["categorical", "real"]
            for j in [1, 3, 5]
            for i in range(1, 4)
        ]
    )

    print(f"{expected_items = }")
    print(f"{found_items = }")

    assert np.all(
        found_items == expected_items
    ), f"{found_items = } != {expected_items = }"


def test_model_files_exists(run_training, project_path):
    model_type_formats = {"categorical": "onnx", "real": "pt"}
    found_items = np.array(
        sorted(list(os.listdir(os.path.join(project_path, "models"))))
    )
    expected_items = np.array(
        sorted(
            [
                f"sequifier-model-{model_type}-{j}-{kind}-3.{model_type_formats[model_type]}"
                for model_type in ["categorical", "real"]
                for j in [1, 3, 5]
                for kind in ["best", "last"]
            ]
            + ["sequifier-model-real-1-best-3-autoregression.onnx"]
        )
    )

    print(f"{expected_items = }")
    print(f"{found_items = }")
    assert np.all(
        found_items == expected_items
    ), f"{found_items = } != {expected_items = }"
