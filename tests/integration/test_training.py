import json
import os

import numpy as np
import pandas as pd
import pytest


def test_checkpoint_files_exists(run_training, project_path):

    found_items = sorted(list(os.listdir(os.path.join(project_path, "checkpoints"))))
    expected_items = np.array([f"model-default-epoch-{i}.pt" for i in range(1, 4)])

    assert np.all(found_items == expected_items), found_items


def test_model_files_exists(run_training, project_path):

    found_items = sorted(list(os.listdir(os.path.join(project_path, "models"))))
    expected_items = np.array(
        ["sequifier-default-best.onnx", "sequifier-default-last.onnx"]
    )

    assert np.all(found_items == expected_items), found_items
