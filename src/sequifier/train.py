import copy
import glob
import math
import os
import re
import time
import uuid
import warnings

import numpy as np
import pandas as pd
import torch
import torch._dynamo
from torch import Tensor, nn
from torch.nn import ModuleDict, TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import one_hot

torch._dynamo.config.suppress_errors = True
from sequifier.config.train_config import load_train_config
from sequifier.helpers import (PANDAS_TO_TORCH_TYPES, LogFile,
                               construct_index_maps, normalize_path,
                               numpy_to_pytorch, read_data,
                               subset_to_selected_columns)


def train(args, args_config):
    config_path = (
        args.config_path if args.config_path is not None else "configs/train.yaml"
    )

    config = load_train_config(config_path, args_config, args.on_unprocessed)

    column_types = {
        col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
        for col in config.column_types
    }

    data_train = read_data(
        normalize_path(config.training_data_path, config.project_path),
        config.read_format,
    )
    check_target_validity(data_train, config.target_columns)
    if config.selected_columns is not None:
        data_train = subset_to_selected_columns(data_train, config.selected_columns)

    X_train, y_train = numpy_to_pytorch(
        data_train,
        column_types,
        config.selected_columns,
        config.target_columns,
        config.seq_length,
        config.training_spec.device,
        to_device=False,
    )
    del data_train

    data_valid = read_data(
        normalize_path(config.validation_data_path, config.project_path),
        config.read_format,
    )
    check_target_validity(data_valid, config.target_columns)
    if config.selected_columns is not None:
        data_valid = subset_to_selected_columns(data_valid, config.selected_columns)

    X_valid, y_valid = numpy_to_pytorch(
        data_valid,
        column_types,
        config.selected_columns,
        config.target_columns,
        config.seq_length,
        config.training_spec.device,
        to_device=False,
    )
    del data_valid

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = torch.compile(TransformerModel(config).to(config.training_spec.device))

    model.train_model(X_train, y_train, X_valid, y_valid)


def check_target_validity(data, target_columns):
    target_column_filter = np.logical_or.reduce(
        [data["inputCol"] == target_column for target_column in target_columns]
    )
    assert np.all(
        np.isnan(data.loc[target_column_filter, "target"].values) == False
    ), "Some target values are NaN"


def format_number(number):
    order_of_magnitude = math.floor(math.log(number, 10))
    number_adjusted = number * (10 ** (-order_of_magnitude))
    return f"{number_adjusted:5.2f}e{order_of_magnitude}"


class TransformerModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.project_path = hparams.project_path
        self.model_type = "Transformer"
        self.model_name = (
            hparams.model_name
            if hparams.model_name is not None
            else uuid.uuid4().hex[:8]
        )

        self.selected_columns = hparams.selected_columns
        self.categorical_columns = [
            col
            for col in hparams.categorical_columns
            if ((self.selected_columns is None) or (col in self.selected_columns))
        ]
        self.real_columns = [
            col
            for col in hparams.real_columns
            if ((self.selected_columns is None) or (col in self.selected_columns))
        ]

        self.target_columns = hparams.target_columns
        self.target_column_types = hparams.target_column_types
        self.loss_weights = hparams.training_spec.loss_weights
        self.seq_length = hparams.seq_length
        self.n_classes = hparams.n_classes
        self.inference_batch_size = hparams.inference_batch_size
        self.log_interval = hparams.training_spec.log_interval
        self.class_share_log_columns = hparams.training_spec.class_share_log_columns
        self.index_maps = construct_index_maps(
            hparams.id_maps, self.class_share_log_columns, True
        )
        self.export_onnx = hparams.export_onnx
        self.export_pt = hparams.export_pt
        self.export_with_dropout = hparams.export_with_dropout
        self.early_stopping_epochs = hparams.training_spec.early_stopping_epochs
        self.hparams = hparams

        self.drop = nn.Dropout(hparams.training_spec.dropout)
        self.encoder = ModuleDict()
        self.pos_encoder = ModuleDict()
        for col, n_classes in self.n_classes.items():
            if col in self.categorical_columns:
                self.encoder[col] = nn.Embedding(n_classes, hparams.model_spec.d_model)
                self.pos_encoder[col] = nn.Embedding(
                    self.seq_length, hparams.model_spec.d_model
                )

        self.real_columns_repetitions = self.get_real_columns_repetitions(
            self.real_columns, hparams.model_spec.nhead
        )

        embedding_size = (
            hparams.model_spec.d_model * len(self.categorical_columns)
        ) + int(np.sum(list(self.real_columns_repetitions.values())))

        encoder_layers = TransformerEncoderLayer(
            embedding_size,
            hparams.model_spec.nhead,
            hparams.model_spec.d_hid,
            hparams.training_spec.dropout,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, hparams.model_spec.nlayers, enable_nested_tensor=False
        )

        self.decoder = ModuleDict()
        for target_column, target_column_type in self.target_column_types.items():
            if target_column_type == "categorical":
                self.decoder[target_column] = nn.Linear(
                    embedding_size,
                    self.n_classes[target_column],
                )
            elif target_column_type == "real":
                self.decoder[target_column] = nn.Linear(embedding_size, 1)
            else:
                raise Exception(
                    f"{target_column_type = } not in ['categorical', 'real']"
                )

        self.criterion = {
            target_column: eval(
                f"torch.nn.{hparams.training_spec.criterion[target_column]}()"
            )
            for target_column in self.target_columns
        }
        self.batch_size = hparams.training_spec.batch_size
        self.accumulation_steps = hparams.training_spec.accumulation_steps
        self.device = hparams.training_spec.device

        self.src_mask = self.generate_square_subsequent_mask(self.seq_length).to(
            self.device
        )

        self.init_weights()
        self.optimizer = self.get_optimizer(
            **self.filter_key(hparams.training_spec.optimizer, "name")
        )
        self.scheduler = self.get_scheduler(
            **self.filter_key(hparams.training_spec.scheduler, "name")
        )

        self.iter_save = hparams.training_spec.iter_save
        self.continue_training = hparams.training_spec.continue_training
        load_string = self.load_weights_conditional()
        self.initialize_log_file()
        self.log_file.write(load_string)

    def get_real_columns_repetitions(self, real_columns, nhead):
        real_columns_repetitions = {col: 1 for col in real_columns}
        column_index = dict(enumerate(real_columns))
        for i in range(nhead * len(real_columns)):
            if np.sum(list(real_columns_repetitions.values())) % nhead != 0:
                j = i % len(real_columns)
                real_columns_repetitions[column_index[j]] += 1
        assert np.sum(list(real_columns_repetitions.values())) % nhead == 0

        return real_columns_repetitions

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

    def filter_key(self, dict_, key):
        return {k: v for k, v in dict_.items() if k != key}

    def init_weights(self) -> None:
        initrange = 0.01
        for col in self.categorical_columns:
            self.encoder[col].weight.data.uniform_(-initrange, initrange)

        for target_column in self.target_columns:
            self.decoder[target_column].bias.data.zero_()
            self.decoder[target_column].weight.data.uniform_(-initrange, initrange)

    def forward_train(self, src: dict[str, Tensor]) -> dict[str, Tensor]:
        srcs = []
        for col in self.categorical_columns:
            src_t = self.encoder[col](src[col].T) * math.sqrt(
                self.hparams.model_spec.d_model
            )
            pos = (
                torch.arange(0, self.seq_length, dtype=torch.long, device=self.device)
                .repeat(src_t.shape[1], 1)
                .T
            )
            src_p = self.pos_encoder[col](pos)

            src_c = self.drop(src_t + src_p)

            srcs.append(src_c)

        for col in self.real_columns:
            srcs.append(
                src[col].T.unsqueeze(2).repeat(1, 1, self.real_columns_repetitions[col])
            )

        src = torch.cat(srcs, 2)

        output = self.transformer_encoder(src, self.src_mask)
        output = {
            target_column: self.decoder[target_column](output)
            for target_column in self.target_columns
        }

        return output

    def forward(self, src: dict[str, Tensor]) -> dict[str, Tensor]:
        output = self.forward_train(src)
        return {target_column: out[-1, :, :] for target_column, out in output.items()}

    def train_model(self, X_train, y_train, X_valid, y_valid):
        best_val_loss = float("inf")
        n_epochs_no_improvemet = 0

        for epoch in range(
            self.start_epoch, self.hparams.training_spec.epochs + self.start_epoch
        ):
            if (
                self.early_stopping_epochs is None
                or n_epochs_no_improvemet < self.early_stopping_epochs
            ):
                epoch_start_time = time.time()
                self.train_epoch(X_train, y_train, epoch)
                total_loss, total_losses, output = self.evaluate(X_valid, y_valid)
                elapsed = time.time() - epoch_start_time
                self.log_file.write("-" * 89)
                self.log_file.write(
                    f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                    f"valid loss {format_number(total_loss)} | baseline loss {format_number(self.baseline_loss)}"
                )

                if len(total_losses) > 1:
                    self.log_file.write(
                        " - ".join(
                            [
                                f"{target_column} loss: {format_number(tloss)}"
                                for target_column, tloss in total_losses.items()
                            ]
                        )
                    )
                    self.log_file.write(
                        " - ".join(
                            [
                                f"{target_column} baseline loss: {format_number(tloss)}"
                                for target_column, tloss in self.baseline_losses.items()
                            ]
                        )
                    )

                for categorical_column in self.class_share_log_columns:
                    output_values = (
                        output[categorical_column].argmax(1).cpu().detach().numpy()
                    )
                    output_counts = pd.Series(output_values).value_counts().sort_index()
                    output_counts = output_counts / output_counts.sum()
                    value_shares = " | ".join(
                        [
                            f"{self.index_maps[categorical_column][value]}: {share:5.5f}"
                            for value, share in output_counts.to_dict().items()
                        ]
                    )
                    self.log_file.write(f"{categorical_column}: {value_shares}")

                self.log_file.write("-" * 89)

                if total_loss < best_val_loss:
                    best_val_loss = total_loss
                    best_model = self.copy_model()

                    n_epochs_no_improvemet = 0
                else:
                    n_epochs_no_improvemet += 1

                self.scheduler.step()
                if (epoch) % self.iter_save == 0:
                    self.save((epoch), total_loss)

                last_epoch = int(epoch)

        self.export(self, "last", last_epoch)
        self.export(best_model, "best", last_epoch)
        self.log_file.write("Training transformer complete")
        self.log_file.close()

    def train_epoch(self, X_train, y_train, epoch) -> None:
        self.train()  # turn on train mode
        total_loss = 0.0
        start_time = time.time()

        num_batches = math.ceil(
            len(X_train[self.target_columns[0]]) / self.batch_size
        )  # any column will do
        batch_order = list(
            np.random.choice(
                np.arange(num_batches), size=num_batches, replace=False
            ).flatten()
        )
        for batch_count, batch in enumerate(batch_order):
            batch_start = batch * self.batch_size

            data, targets = self.get_batch(
                X_train, y_train, batch_start, self.batch_size, to_device=True
            )
            output = self.forward_train(data)

            loss, losses = self.calculate_loss(output, targets)

            with torch.no_grad():
                torch.cuda.empty_cache()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

            if (
                self.accumulation_steps is None
                or (batch_count // self.batch_size + 1) % self.accumulation_steps == 0
            ):
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            if batch_count % self.log_interval == 0 and batch_count > 0:
                lr = self.scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / self.log_interval
                cur_loss_normalized = (
                    1000 * total_loss / (self.log_interval * self.batch_size)
                )
                self.log_file.write(
                    f"| epoch {epoch:3d} | {batch_count:5d}/{num_batches:5d} batches | "
                    f"lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss_normalized :5.5f}"
                )
                total_loss = 0.0
                start_time = time.time()

    def calculate_loss(self, output, targets):
        losses = {}
        for target_column, target_column_type in self.target_column_types.items():
            if target_column_type == "categorical":
                output[target_column] = output[target_column].reshape(
                    -1, self.n_classes[target_column]
                )
            elif target_column_type == "real":
                try:
                    output[target_column] = output[target_column].reshape(-1)
                except:
                    import code

                    code.interact(local=locals())
            else:
                pass

            losses[target_column] = self.criterion[target_column](
                output[target_column], targets[target_column].T.contiguous().reshape(-1)
            )

        loss = None
        for target_column in losses.keys():
            losses[target_column] = losses[target_column] * (
                self.loss_weights[target_column]
                if self.loss_weights is not None
                else 1.0
            )
            if loss is None:
                loss = losses[target_column]
            else:
                loss += losses[target_column]
        return (loss, losses)

    def copy_model(self):
        log_file = self.log_file
        del self.log_file
        model_copy = copy.deepcopy(self)
        self.log_file = log_file
        return model_copy

    def _transform_val(self, col, val):
        if self.target_column_types[col] == "categorical":
            return (
                one_hot(val, self.n_classes[col])
                .reshape(-1, self.n_classes[col])
                .float()
            )
        else:
            assert self.target_column_types[col] == "real"
            return val

    def evaluate(self, X_valid, y_valid) -> float:
        self.eval()  # turn on evaluation mode

        with torch.no_grad():
            data, targets = self.get_batch(
                X_valid, y_valid, 0, self.batch_size, to_device=True
            )
            output = self.forward_train(data)
            total_loss, total_losses = self.calculate_loss(output, targets)

            torch.cuda.empty_cache()

        denominator = X_valid[self.target_columns[0]].size(0)  # any column will do
        total_loss = total_loss / denominator
        total_losses = {
            target_column: tloss / denominator
            for target_column, tloss in total_losses.items()
        }

        if not hasattr(self, "baseline_loss"):
            # import code; code.interact(local=locals())
            self.baseline_loss, self.baseline_losses = self.calculate_loss(
                {
                    col: self._transform_val(col, val[:, :-1])
                    for col, val in targets.items()
                },  # this variant is chosen because the same batch might have several "sequenceId" sequences
                {col: val[:, 1:] for col, val in targets.items()},
            )
            shape_1_adjustment = self.seq_length / (self.seq_length - 1)
            self.baseline_loss = (self.baseline_loss / denominator) * shape_1_adjustment
            self.baseline_losses = {
                target_column: (bloss / denominator) * shape_1_adjustment
                for target_column, bloss in total_losses.items()
            }

        return total_loss, total_losses, output

    def get_batch(self, X, y, batch_start, batch_size, to_device):
        if to_device:
            return (
                {
                    col: X[col][batch_start : batch_start + batch_size, :].to(
                        self.device
                    )
                    for col in X.keys()
                },
                {
                    target_column: y[target_column][
                        batch_start : batch_start + batch_size, :
                    ].to(self.device)
                    for target_column in y.keys()
                },
            )
        else:
            return (
                {
                    col: X[col][batch_start : batch_start + batch_size, :]
                    for col in X.keys()
                },
                {
                    target_column: y[target_column][
                        batch_start : batch_start + batch_size, :
                    ]
                    for target_column in y.keys()
                },
            )

    def export(self, model, suffix, epoch):
        self.eval()

        os.makedirs(os.path.join(self.project_path, "models"), exist_ok=True)
        if self.export_onnx:
            x_cat = {
                col: torch.randint(
                    0,
                    self.n_classes[col],
                    (self.inference_batch_size, self.seq_length),
                ).to(self.device)
                for col in self.categorical_columns
            }
            x_real = {
                col: torch.rand(self.inference_batch_size, self.seq_length).to(
                    self.device
                )
                for col in self.real_columns
            }

            x = {"src": {**x_cat, **x_real}}

            # Export the model
            export_path = os.path.join(
                self.project_path,
                "models",
                f"sequifier-{self.model_name}-{suffix}-{epoch}.onnx",
            )
            training_mode = (
                torch._C._onnx.TrainingMode.TRAINING
                if self.export_with_dropout
                else torch._C._onnx.TrainingMode.EVAL
            )
            torch.onnx.export(
                model,  # model being run
                x,  # model input (or a tuple for multiple inputs)
                export_path,  # where to save the model (can be a file or file-like object)
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=14,  # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names=["input"],  # the model's input names
                output_names=["output"],  # the model's output names
                dynamic_axes={
                    "input": {0: "batch_size"},  # variable length axes
                    "output": {0: "batch_size"},
                },
                training=training_mode,
            )
        if self.export_pt:
            export_path = os.path.join(
                self.project_path,
                "models",
                f"sequifier-{self.model_name}-{suffix}-{epoch}.pt",
            )
            torch.save(
                {
                    "model_state_dict": self.state_dict(),
                    "export_with_dropout": self.export_with_dropout,
                },
                export_path,
            )

    def save(self, epoch, val_loss):
        os.makedirs(os.path.join(self.project_path, "checkpoints"), exist_ok=True)

        output_path = os.path.join(
            self.project_path,
            "checkpoints",
            f"{self.model_name}-epoch-{epoch}.pt",
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": val_loss,
            },
            output_path,
        )
        self.log_file.write(f"Saved model to {output_path}")

    def get_optimizer(self, **kwargs):
        optimizer_class = eval(
            f"torch.optim.{self.hparams.training_spec.optimizer.name}"
        )
        return optimizer_class(
            self.parameters(), lr=self.hparams.training_spec.lr, **kwargs
        )

    def get_scheduler(self, **kwargs):
        scheduler_class = eval(
            f"torch.optim.lr_scheduler.{self.hparams.training_spec.scheduler.name}"
        )
        return scheduler_class(self.optimizer, **kwargs)

    def initialize_log_file(self):
        os.makedirs(os.path.join(self.project_path, "logs"), exist_ok=True)
        open_mode = "w" if self.start_epoch == 1 else "a"
        self.log_file = LogFile(
            os.path.join(self.project_path, "logs", f"sequifier-{self.model_name}.txt"),
            open_mode,
        )

    def load_weights_conditional(self):
        latest_model_path = self.get_latest_model_name()

        if latest_model_path is not None and self.continue_training:
            checkpoint = torch.load(
                latest_model_path, map_location=torch.device(self.device)
            )
            self.load_state_dict(checkpoint["model_state_dict"])
            self.start_epoch = (
                int(re.findall("epoch-([0-9]+)", latest_model_path)[0]) + 1
            )
            return f"Loading model weights from {latest_model_path}"
        else:
            self.start_epoch = 1
            return "Initializing new model"

    def get_latest_model_name(self):
        checkpoint_path = os.path.join(self.project_path, "checkpoints", "*")

        files = glob.glob(
            checkpoint_path
        )  # * means all if need specific format then *.csv
        files = [
            file for file in files if os.path.split(file)[1].startswith(self.model_name)
        ]
        if len(files):
            return max(files, key=os.path.getctime)
        else:
            return None


def load_inference_model(
    model_path, training_config_path, args_config, device, infer_with_dropout
):
    training_config = load_train_config(
        training_config_path, args_config, args_config["on_unprocessed"]
    )

    with torch.no_grad():
        model = TransformerModel(training_config)
        model.log_file.write(f"Loading model weights from {model_path}")
        model_state = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(model_state["model_state_dict"])

        model.eval()

        if infer_with_dropout:
            if not model_state["export_with_dropout"]:
                warnings.warn(
                    "Model was exported with 'export_with_dropout'==False. By setting 'infer_with_dropout' to True, you are overriding this configuration"
                )
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()

        model = torch.compile(model).to(device)

    return model


def infer_with_model(model, x, device, size, target_columns):
    outs0 = [
        model.forward(
            {col: torch.from_numpy(x_).to(device) for col, x_ in x_sub.items()}
        )
        for x_sub in x
    ]
    outs = {
        target_column: np.concatenate(
            [o[target_column].cpu().detach().numpy() for o in outs0],
            axis=0,
        )[:size, :]
        for target_column in target_columns
    }

    return outs
