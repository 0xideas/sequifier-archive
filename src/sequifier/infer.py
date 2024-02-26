import json
import os

import numpy as np
import onnxruntime
import pandas as pd

from sequifier.config.infer_config import load_inferer_config
from sequifier.helpers import PANDAS_TO_TORCH_TYPES, numpy_to_pytorch


class Inferer(object):
    def __init__(
        self,
        model_path,
        project_path,
        id_map,
        map_to_id,
        categorical_columns,
        real_columns,
        target_column,
        target_column_type,
        batch_size,
    ):
        if target_column_type == "categorical":
            self.index_map = (
                {v: k for k, v in id_map[target_column].items()} if map_to_id else None
            )

        self.map_to_id = map_to_id
        model_path_load = os.path.join(project_path, model_path)
        self.ort_session = onnxruntime.InferenceSession(model_path_load)
        self.categorical_columns = categorical_columns
        self.real_columns = real_columns
        self.target_column = target_column
        self.target_column_type = target_column_type
        self.batch_size = batch_size

    def adjust_x_size(self, x):
        size = x[self.target_column].shape[0]
        if size == self.batch_size:
            return [x]
        elif size < self.batch_size:
            x_expanded = {
                col: self.expand_to_batch_size(x_col) for col, x_col in x.items()
            }
            return [x_expanded]
        else:
            starts = range(0, size, self.batch_size)
            ends = range(self.batch_size, size + self.batch_size, self.batch_size)
            xs = [
                {col: x_col[start:end, :] for col, x_col in x.items()}
                for start, end in zip(starts, ends)
            ]
            return xs

    def infer_probs_any_size(self, x):
        size = x[self.target_column].shape[0]
        x_adjusted = self.adjust_x_size(x)
        return np.concatenate([self.infer_probs(x_sub) for x_sub in x_adjusted], 0)[
            :size, :
        ]

    def infer_real_any_size(self, x):
        # import code; code.interact(local=locals())
        size = x[self.target_column].shape[0]
        x_adjusted = self.adjust_x_size(x)
        return np.concatenate([self.infer_pure(x_sub) for x_sub in x_adjusted], 0)[
            :size, :
        ]

    def expand_to_batch_size(self, x):
        filler = self.batch_size - x.shape[0]
        return np.concatenate([x, x[0:filler, :]], axis=0)

    def infer_pure(self, x):
        ort_inputs = {
            session_input.name: self.expand_to_batch_size(x[col])
            for session_input, col in zip(
                self.ort_session.get_inputs(),
                self.categorical_columns + self.real_columns,
            )
        }
        ort_outs = self.ort_session.run(None, ort_inputs)[0]

        return ort_outs

    def infer_probs(self, x):
        ort_outs = self.infer_pure(x)

        normalizer = np.repeat(
            np.sum(np.exp(ort_outs), axis=1), ort_outs.shape[1]
        ).reshape(ort_outs.shape)
        probs = np.exp(ort_outs) / normalizer
        return probs

    def infer_categorical(self, x, probs=None):
        if probs is None:
            probs = self.infer_probs_any_size(x)
        preds = probs.argmax(1)
        if self.map_to_id:
            preds = np.array([self.index_map[i] for i in preds])
        return preds

    def infer(self, x, probs=None):
        if self.target_column_type == "categorical":
            return self.infer_categorical(x, probs)
        elif self.target_column_type == "real":
            return self.infer_real_any_size(x)
        else:
            pass


def infer(args, args_config):
    config = load_inferer_config(args.config_path, args_config, args.on_preprocessed)

    column_types = {
        col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
        for col in config.column_types
    }

    model_id = os.path.split(config.model_path)[1].replace(".onnx", "")

    print(f"Inferring for {model_id}")

    inference_data_path = os.path.join(config.project_path, config.inference_data_path)

    data = pd.read_csv(inference_data_path, sep=",", decimal=".", index_col=None)
    X, _ = numpy_to_pytorch(
        data, column_types, config.target_column, config.seq_length, config.device
    )
    X = {col: X_col.detach().cpu().numpy() for col, X_col in X.items()}
    del data

    if config.map_to_id:
        assert (
            config.ddconfig_path is not None
        ), "If you want to map to id, you need to provide a file path to a json that contains: {{'id_map':{...}}} to ddconfig_path"
        with open(os.path.join(config.project_path, config.ddconfig_path), "r") as f:
            id_maps = json.loads(f.read())["id_maps"]
    else:
        id_maps = None

    inferer = Inferer(
        config.model_path,
        config.project_path,
        id_maps,
        config.map_to_id,
        config.categorical_columns,
        config.real_columns,
        config.target_column,
        config.target_column_type,
        config.batch_size,
    )

    if config.output_probabilities:
        probs = inferer.infer_probs_any_size(X)
        os.makedirs(
            os.path.join(config.project_path, "outputs", "probabilities"), exist_ok=True
        )
        probabilities_path = os.path.join(
            config.project_path,
            "outputs",
            "probabilities",
            f"{model_id}_probabilities.csv",
        )
        print(f"Writing probabilities to {probabilities_path}")
        pd.DataFrame(probs).to_csv(
            probabilities_path, sep=",", decimal=".", index=False
        )
        preds = inferer.infer(None, probs)
    else:
        preds = inferer.infer(X)

    os.makedirs(
        os.path.join(config.project_path, "outputs", "predictions"), exist_ok=True
    )
    predictions_path = os.path.join(
        config.project_path, "outputs", "predictions", f"{model_id}_predictions.csv"
    )

    print(f"Writing predictions to {predictions_path}")
    pd.DataFrame(preds).to_csv(predictions_path, sep=",", decimal=".", index=False)
    print("Inference complete")
