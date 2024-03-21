import json
import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def predictions(run_inference, project_path):
    preds = {"categorical": {}, "real": {}}
    model_names = [
        f"model-{variant}-{model_number}-best-3"
        for variant in ["categorical", "real"]
        for model_number in [1, 3, 5]
    ]
    model_names.append("model-real-1-best-3-autoregression")
    for model_name in model_names:
        prediction_path = os.path.join(
            project_path,
            "outputs",
            "predictions",
            f"sequifier-{model_name}_predictions.csv",
        )
        variant = model_name.split("-")[1]
        preds[variant][model_name] = pd.read_csv(
            prediction_path, sep=",", decimal=".", index_col=None
        ).values.flatten()

    return preds


@pytest.fixture()
def probabilities(run_inference, project_path):
    probs = {}
    for model_number in [1, 3, 5]:
        model_name = f"model-categorical-{model_number}"
        prediction_path = os.path.join(
            project_path,
            "outputs",
            "probabilities",
            f"sequifier-{model_name}-best-3_probabilities.csv",
        )
        probs[model_name] = pd.read_csv(
            prediction_path, sep=",", decimal=".", index_col=None
        )
    return probs


def test_predictions_real(predictions):
    for model_name, model_predictions in predictions["real"].items():
        assert np.all([v > -10.0 and v < 10.0 for v in model_predictions])


def test_predictions_cat(predictions):
    valid_values = np.arange(100, 130)
    for model_name, model_predictions in predictions["categorical"].items():
        assert np.all([v in valid_values for v in model_predictions])


def test_probabilities(probabilities):
    for model_name, model_probabilities in probabilities.items():
        assert model_probabilities.shape[1] == 31

        np.testing.assert_almost_equal(
            model_probabilities.sum(1), np.ones(model_probabilities.shape[0]), decimal=5
        )
