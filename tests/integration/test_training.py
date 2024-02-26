import json
import os

import numpy as np
import pandas as pd
import pytest


def test_checkpoint_files_exists(run_training, project_path):

    found_items = sorted(list(os.listdir(os.path.join(project_path, "checkpoints"))))
    expected_items = np.array(sum([[f"model-model-{j}-epoch-{i}.pt" for i in range(1, 4)] for j in [1, 3, 5]], []))

    assert np.all(found_items == expected_items), f"{found_items = } != {expected_items = }"


def test_model_files_exists(run_training, project_path):

    found_items = sorted(list(os.listdir(os.path.join(project_path, "models"))))
    expected_items = [
        [f"sequifier-model-{j}-best.onnx", f"sequifier-model-{j}-last.onnx"] for j in [1, 3, 5]
    ]
    expected_items = np.array(sum(expected_items, []))
    
    assert np.all(found_items == expected_items), f"{found_items = } != {expected_items = }"
