import json
import os
import warnings

import numpy as np
import onnxruntime
import pandas as pd
import torch

from sequifier.config.infer_config import load_inferer_config
from sequifier.helpers import (
    PANDAS_TO_TORCH_TYPES,
    numpy_to_pytorch,
    read_data,
    subset_to_selected_columns,
    write_data,
)
from sequifier.train import infer_with_model, load_inference_model


def infer(args, args_config):
    config_path = (
        args.config_path if args.config_path is not None else "configs/infer.yaml"
    )

    config = load_inferer_config(config_path, args_config, args.on_unprocessed)

    if config.map_to_id:
        assert (
            config.ddconfig_path is not None
        ), "If you want to map to id, you need to provide a file path to a json that contains: {{'id_map':{...}}} to ddconfig_path"
        with open(os.path.join(config.project_path, config.ddconfig_path), "r") as f:
            id_maps = json.loads(f.read())["id_maps"]
    else:
        id_maps = None

    inferer = Inferer(
        config.inference_model_path,
        config.project_path,
        id_maps,
        config.map_to_id,
        config.categorical_columns,
        config.real_columns,
        config.target_column,
        config.target_column_type,
        config.sample_from_distribution,
        config.infer_with_dropout,
        config.inference_batch_size,
        config.device,
        args_config=args_config,
        training_config_path=config.training_config_path,
    )

    column_types = {
        col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
        for col in config.column_types
    }

    model_id = os.path.split(config.inference_model_path)[1].replace(
        f".{inferer.inference_model_type}", ""
    )

    print(f"Inferring for {model_id}")

    inference_data_path = os.path.join(config.project_path, config.inference_data_path)

    data = read_data(inference_data_path, config.read_format)
    if config.selected_columns is not None:
        data = subset_to_selected_columns(data, config.selected_columns)

    if not config.auto_regression:
        probs, preds = get_probs_preds(config, inferer, data, column_types)
    else:
        probs, preds = get_probs_preds_auto_regression(
            config, inferer, data, column_types
        )

    os.makedirs(
        os.path.join(config.project_path, "outputs", "predictions"), exist_ok=True
    )

    if config.output_probabilities:
        os.makedirs(
            os.path.join(config.project_path, "outputs", "probabilities"), exist_ok=True
        )
        probabilities_path = os.path.join(
            config.project_path,
            "outputs",
            "probabilities",
            f"{model_id}_probabilities.{config.write_format}",
        )
        print(f"Writing probabilities to {probabilities_path}")
        write_data(pd.DataFrame(probs), probabilities_path, config.write_format)

    predictions_path = os.path.join(
        config.project_path,
        "outputs",
        "predictions",
        f"{model_id}_predictions.{config.write_format}",
    )

    print(f"Writing predictions to {predictions_path}")
    write_data(
        pd.DataFrame(preds, columns=["model_output"]),
        predictions_path,
        config.write_format,
    )
    print("Inference complete")


def get_probs_preds(config, inferer, data, column_types):
    X, _ = numpy_to_pytorch(
        data,
        column_types,
        config.target_column,
        config.seq_length,
        config.device,
        to_device=False,
    )
    X = {col: X_col.numpy() for col, X_col in X.items()}

    del data

    if config.output_probabilities:
        probs = inferer.infer_probs_any_size(X)
        preds = inferer.infer(None, probs)
    else:
        probs = None
        preds = inferer.infer(X)

    return (probs, preds)


def get_probs_preds_auto_regression(config, inferer, data, column_types):
    data = data.sort_values(["subsequenceId", "sequenceId"])
    n_input_col_values = len(np.unique(data["inputCol"]))
    preds_list, probs_list, indices = [], [], []
    subsequence_ids = sorted(list(np.unique(data["subsequenceId"])))
    for subsequence_id in subsequence_ids:
        data_subset = data.loc[data["subsequenceId"] == subsequence_id, :]
        indices.append(
            np.array(data_subset.index)[
                np.arange(0, data_subset.shape[0], n_input_col_values)
            ]
        )
        probs, preds = get_probs_preds(config, inferer, data_subset, column_types)
        preds_list.append(preds)
        if probs is not None:
            probs_list.append(probs)

        for offset in range(1, int(list(data.columns)[3])):
            target_subsequence_filter = data["subsequenceId"].values == (
                subsequence_id + offset
            )  # filter all data on target subsequence id
            data_col_filter = (
                data["inputCol"].values == config.target_column
            )  # filter on the target column
            data_subset_sequence_ids = sorted(
                list(np.unique(data_subset["sequenceId"]))
            )
            sequence_filter = np.array(
                [
                    sequence_id in data_subset_sequence_ids
                    for sequence_id in data["sequenceId"]
                ]
            )
            f = np.logical_and.reduce(
                [
                    target_subsequence_filter,
                    data_col_filter,
                    sequence_filter,
                ]
            )  # filter on target subsequence id and target column and sequence ids in data_subset

            if np.sum(f) > 0:
                f_sequence_ids = sorted(
                    list(np.unique(data.loc[f, "sequenceId"]))
                )  # sequence ids that exist in those rows

                f_sequence_ids_filter = np.array(
                    [
                        sequence_id in f_sequence_ids
                        for sequence_id in data_subset["sequenceId"]
                    ]
                )  # subset data_subset to those rows with sequence ids that also exist for the target subsequence id
                data_subset_col_filter = (
                    data_subset["inputCol"].values == config.target_column
                )  # filter data_subset to target column
                f_sequence_ids_filter_subset = np.logical_and(
                    f_sequence_ids_filter, data_subset_col_filter
                )  # combine

                f_preds = preds[
                    f_sequence_ids_filter[
                        np.arange(0, len(f_sequence_ids_filter), n_input_col_values)
                    ]
                ]  # subset preds to those sequence ids that exist for target subsequence id
                # f_sequence_ids_filter has to be subset itself because it contains n_input_col_values
                # rows for each observtion

                f_data_subset = data_subset.loc[
                    f_sequence_ids_filter_subset, ["sequenceId", "subsequenceId"]
                ]  # find sequence ids and subsequence ids that exist for both the original subsequence
                # id and the target subsequence id
                assert (
                    data.loc[f, str(offset)].shape[0] == f_preds.shape[0]
                ), f"{data.columns = }: {data.loc[f,:].values = }  != {f_preds.shape = }"
                assert np.all(
                    data.loc[f, "sequenceId"].values
                    == f_data_subset["sequenceId"].values
                )
                assert np.all(
                    (f_data_subset["subsequenceId"].values + 1) == (subsequence_id + 1)
                ), f"{f_data_subset['subsequenceId'].values + 1} != {(subsequence_id + 1)}"
                data.loc[f, str(offset)] = f_preds

    preds = np.concatenate(preds_list, axis=0)
    index_order = np.concatenate(indices, axis=0)
    assert len(preds) == len(index_order)
    preds = np.array(
        [p for p, i in sorted(list(zip(preds, index_order)), key=lambda t: t[1])]
    )
    if len(probs_list):
        probs = np.concatenate(probs, axis=0)
        assert probs.shape[0] == len(index_order)
        probs = np.array(
            [p for p, i in sorted(list(zip(probs, index_order)), key=lambda t: t[1])]
        )

    else:
        probs = None
    return (probs, preds)


class Inferer(object):
    def __init__(
        self,
        inference_model_path,
        project_path,
        id_map,
        map_to_id,
        categorical_columns,
        real_columns,
        target_column,
        target_column_type,
        sample_from_distribution,
        infer_with_dropout,
        inference_batch_size,
        device,
        args_config,
        training_config_path,
    ):
        if target_column_type == "categorical":
            self.index_map = (
                {v: k for k, v in id_map[target_column].items()} if map_to_id else None
            )
            if isinstance(list(self.index_map.values())[0], str):
                self.index_map[0] = "unknown"
            else:
                self.index_map[0] = np.min(self.index_map.values()) - 1

        self.map_to_id = map_to_id
        self.device = device
        self.categorical_columns = categorical_columns
        self.real_columns = real_columns
        self.target_column = target_column
        self.target_column_type = target_column_type
        self.sample_from_distribution = sample_from_distribution
        self.infer_with_dropout = infer_with_dropout
        self.inference_batch_size = inference_batch_size
        self.inference_model_type = inference_model_path.split(".")[-1]
        self.inference_model_path_load = os.path.join(
            project_path, inference_model_path
        )
        self.args_config = args_config
        self.training_config_path = training_config_path

        if self.inference_model_type == "onnx":
            execution_providers = [
                "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
            ]
            kwargs = {}
            if self.infer_with_dropout:
                kwargs["disabled_optimizers"] = ["EliminateDropout"]

                warnings.warn(
                    "For inference with onnx, 'infer_with_dropout==True' is only effective if 'export_with_dropout==True' in training"
                )

            self.ort_session = onnxruntime.InferenceSession(
                self.inference_model_path_load, providers=execution_providers, **kwargs
            )
        if self.inference_model_type == "pt":
            self.inference_model = load_inference_model(
                self.inference_model_path_load,
                self.training_config_path,
                self.args_config,
                self.device,
                self.infer_with_dropout,
            )

    def infer(self, x, probs=None):
        if self.target_column_type == "categorical":
            return self.infer_categorical_any_size(x, probs)
        if self.target_column_type == "real":
            return self.infer_real_any_size(x)

    def infer_categorical_any_size(self, x, probs=None):
        if probs is None:
            probs = self.infer_probs_any_size(x)
        if self.sample_from_distribution is False:
            preds = probs.argmax(1)
        else:
            preds = sample_with_cumsum(probs)
        if self.map_to_id:
            preds = np.array([self.index_map[i] for i in preds])
        return preds

    def infer_probs_any_size(self, x):
        size = x[self.target_column].shape[0]
        if self.inference_model_type == "onnx":
            x_adjusted = self.prepare_inference_batches(x, pad_to_batch_size=True)
            logits = np.concatenate(
                [self.infer_pure(x_sub) for x_sub in x_adjusted], 0
            )[:size, :]
        if self.inference_model_type == "pt":
            x_adjusted = self.prepare_inference_batches(x, pad_to_batch_size=False)
            logits = infer_with_model(
                self.inference_model,
                x_adjusted,
                self.device,
            )
        return normalize(logits)

    def infer_real_any_size(self, x):
        size = x[self.target_column].shape[0]
        if self.inference_model_type == "onnx":
            x_adjusted = self.prepare_inference_batches(x, pad_to_batch_size=True)
            preds = np.concatenate([self.infer_pure(x_sub) for x_sub in x_adjusted], 0)[
                :size, :
            ]
        if self.inference_model_type == "pt":
            x_adjusted = self.prepare_inference_batches(x, pad_to_batch_size=False)
            preds = infer_with_model(
                self.inference_model,
                x_adjusted,
                self.device,
            )
        return preds.flatten()

    def prepare_inference_batches(self, x, pad_to_batch_size):
        size = x[self.target_column].shape[0]
        if size == self.inference_batch_size:
            return [x]
        elif size < self.inference_batch_size:
            if pad_to_batch_size:
                x_expanded = {
                    col: self.expand_to_batch_size(x_col) for col, x_col in x.items()
                }
                return [x_expanded]
            else:
                return [x]
        else:
            starts = range(0, size, self.inference_batch_size)
            ends = range(
                self.inference_batch_size,
                size + self.inference_batch_size,
                self.inference_batch_size,
            )
            xs = [
                {col: x_col[start:end, :] for col, x_col in x.items()}
                for start, end in zip(starts, ends)
            ]
            return xs

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

    def expand_to_batch_size(self, x):
        repetitions = self.inference_batch_size // x.shape[0]
        filler = self.inference_batch_size % x.shape[0]
        return np.concatenate(([x] * repetitions) + [x[0:filler, :]], axis=0)


def normalize(outs):

    normalizer = np.repeat(np.sum(np.exp(outs), axis=1), outs.shape[1]).reshape(
        outs.shape
    )
    probs = np.exp(outs) / normalizer
    return probs


def sample_with_cumsum(probs):
    cumulative_probs = np.cumsum(probs, axis=1)
    random_threshold = np.random.rand(cumulative_probs.shape[0], 1)
    random_threshold = np.repeat(random_threshold, probs.shape[1], axis=1)
    return (random_threshold < cumulative_probs).argmax(axis=1)
