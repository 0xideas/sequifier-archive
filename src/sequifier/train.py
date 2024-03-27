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
from torch import Tensor, nn
from torch.nn import ModuleDict, TransformerEncoder, TransformerEncoderLayer

from sequifier.config.train_config import load_transformer_config
from sequifier.helpers import (PANDAS_TO_TORCH_TYPES, LogFile,
                               numpy_to_pytorch, read_data,
                               subset_to_selected_columns)


class TransformerModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.project_path = hparams.project_path
        self.target_column = hparams.target_column
        self.target_column_type = hparams.target_column_type
        self.selected_columns = hparams.selected_columns
        self.real_columns = [
            col
            for col in hparams.real_columns
            if ((self.selected_columns is None) or (col in self.selected_columns))
        ]
        self.categorical_columns = [
            col
            for col in hparams.categorical_columns
            if ((self.selected_columns is None) or (col in self.selected_columns))
        ]

        self.inference_batch_size = hparams.inference_batch_size
        self.model_name = (
            hparams.model_name
            if hparams.model_name is not None
            else uuid.uuid4().hex[:8]
        )
        self.hparams = hparams
        self.real_columns_repetitions = self.get_real_columns_repetitions(
            self.real_columns, hparams.model_spec.nhead
        )
        self.early_stopping_epochs = hparams.training_spec.early_stopping_epochs
        self.export_with_dropout = hparams.export_with_dropout
        self.export_onnx = hparams.export_onnx
        self.export_pt = hparams.export_pt
        self.model_type = "Transformer"
        self.log_interval = hparams.log_interval
        self.encoder = ModuleDict()
        self.pos_encoder = ModuleDict()
        for col, n_classes in hparams.n_classes.items():
            if col in self.categorical_columns:
                self.encoder[col] = nn.Embedding(n_classes, hparams.model_spec.d_model)
                self.pos_encoder[col] = PositionalEncoding(
                    hparams.model_spec.d_model, hparams.training_spec.dropout
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

        if self.target_column_type == "categorical":
            self.decoder = nn.Linear(
                embedding_size * hparams.seq_length,
                hparams.n_classes[self.target_column],
            )
        elif self.target_column_type == "real":
            self.decoder = nn.Linear(embedding_size * hparams.seq_length, 1)
        else:
            raise Exception(
                f"{self.target_column_type = } not in ['categorical', 'real']"
            )

        self.criterion = eval(f"torch.nn.{hparams.training_spec.criterion}()")
        self.batch_size = hparams.training_spec.batch_size
        self.accumulation_steps = hparams.training_spec.accumulation_steps
        self.device = hparams.training_spec.device

        self.src_mask = self.generate_square_subsequent_mask(
            self.hparams.seq_length
        ).to(self.device)

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

    def initialize_log_file(self):
        os.makedirs(os.path.join(self.project_path, "logs"), exist_ok=True)
        open_mode = "w" if self.start_epoch == 1 else "a"
        self.log_file = LogFile(
            os.path.join(self.project_path, "logs", f"sequifier-{self.model_name}.txt"),
            open_mode,
        )

    def get_real_columns_repetitions(self, real_columns, nhead):
        real_columns_repetitions = {col: 1 for col in real_columns}
        column_index = dict(enumerate(real_columns))
        for i in range(nhead * len(real_columns)):
            if np.sum(list(real_columns_repetitions.values())) % nhead != 0:
                j = i % len(real_columns)
                real_columns_repetitions[column_index[j]] += 1
        assert np.sum(list(real_columns_repetitions.values())) % nhead == 0

        return real_columns_repetitions

    def filter_key(self, dict_, key):
        return {k: v for k, v in dict_.items() if k != key}

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

    def init_weights(self) -> None:
        initrange = 0.1
        for col in self.categorical_columns:
            self.encoder[col].weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: dict[str, Tensor]) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
        Returns:
            output Tensor of shape [batch_size, n_classes]
        """

        srcs = []
        for col in self.categorical_columns:
            src_t = self.encoder[col](src[col].T) * math.sqrt(
                self.hparams.model_spec.d_model
            )
            src_t = self.pos_encoder[col](src_t)
            srcs.append(src_t)

        for col in self.real_columns:
            srcs.append(
                src[col].T.unsqueeze(2).repeat(1, 1, self.real_columns_repetitions[col])
            )

        src = torch.cat(srcs, 2)

        output = self.transformer_encoder(src, self.src_mask)
        transposed = output.transpose(0, 1)
        concatenated = transposed.reshape(
            transposed.size()[0], transposed.size()[1] * transposed.size()[2]
        )
        output = self.decoder(concatenated)
        return output

    def get_batch(self, X, y, batch_start, batch_size, to_device):
        if to_device:
            return (
                {
                    col: X[col][batch_start : batch_start + batch_size, :].to(
                        self.device
                    )
                    for col in X.keys()
                },
                y[batch_start : batch_start + batch_size].to(self.device),
            )
        else:
            return (
                {
                    col: X[col][batch_start : batch_start + batch_size, :]
                    for col in X.keys()
                },
                y[batch_start : batch_start + batch_size],
            )

    def train_epoch(self, X_train, y_train, epoch) -> None:
        self.train()  # turn on train mode
        total_loss = 0.0
        start_time = time.time()

        num_batches = math.ceil(len(X_train[self.target_column]) / self.batch_size)
        batch_order = list(
            np.random.choice(
                np.arange(num_batches), size=num_batches, replace=False
            ).flatten()
        )
        for batch_count, batch in enumerate(batch_order):
            batch_start = batch * self.batch_size
            data, targets = self.get_batch(
                X_train, y_train, batch_start, self.batch_size, to_device=False
            )
            output = self(data)
            if self.target_column_type == "categorical":
                output = output.view(-1, self.hparams.n_classes[self.target_column])
            elif self.target_column_type == "real":
                output = output.flatten()
            else:
                pass

            loss = self.criterion(output, targets)
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
                ppl = math.exp(cur_loss_normalized)
                self.log_file.write(
                    f"| epoch {epoch:3d} | {batch_count:5d}/{num_batches:5d} batches | "
                    f"lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss_normalized :5.5f} | ppl {ppl:8.2f}"
                )
                total_loss = 0.0
                start_time = time.time()

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
                val_loss_normalized = 1000 * self.evaluate(X_valid, y_valid)
                val_ppl = math.exp(val_loss_normalized)
                elapsed = time.time() - epoch_start_time
                self.log_file.write("-" * 89)
                self.log_file.write(
                    f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                    f"valid loss {val_loss_normalized:5.5f} | valid ppl {val_ppl:8.2f}"
                )
                self.log_file.write("-" * 89)

                if val_loss_normalized < best_val_loss:
                    best_val_loss = val_loss_normalized
                    best_model = self.copy_model()

                    n_epochs_no_improvemet = 0
                else:
                    n_epochs_no_improvemet += 1

                self.scheduler.step()
                if epoch % self.iter_save == 0:
                    self.save(epoch, val_loss_normalized)
                last_epoch = int(epoch)

        self.export(self, "last", last_epoch)
        self.export(best_model, "best", last_epoch)
        self.log_file.write("Training transformer complete")
        self.log_file.close()

    def copy_model(self):
        log_file = self.log_file
        del self.log_file
        model_copy = copy.deepcopy(self)
        self.log_file = log_file
        return model_copy

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

    def evaluate(self, X_valid, y_valid) -> float:
        self.eval()  # turn on evaluation mode
        total_loss = 0.0
        with torch.no_grad():
            for batch_start in range(
                0, X_valid[self.target_column].size(0), self.batch_size
            ):
                data, targets = self.get_batch(
                    X_valid, y_valid, batch_start, self.batch_size, to_device=False
                )
                output = self(data)
                if self.target_column_type == "categorical":
                    output = output.view(-1, self.hparams.n_classes[self.target_column])
                elif self.target_column_type == "real":
                    output = output.flatten()
                else:
                    pass

                total_loss += self.criterion(output, targets).item()

        return total_loss / (X_valid[self.target_column].size(0))

    def export(self, model, suffix, epoch):
        self.eval()

        os.makedirs(os.path.join(self.project_path, "models"), exist_ok=True)
        if self.export_onnx:
            x_cat = {
                col: torch.randint(
                    0,
                    self.hparams.n_classes[col],
                    (self.inference_batch_size, self.hparams.seq_length),
                ).to(self.device)
                for col in self.categorical_columns
            }
            x_real = {
                col: torch.rand(self.inference_batch_size, self.hparams.seq_length).to(
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

    def load_weights_conditional(self):

        latest_model_path = self.get_latest_model_name()

        if latest_model_path is not None and self.continue_training:
            checkpoint = torch.load(latest_model_path)
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


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def train(args, args_config):
    config_path = (
        args.config_path if args.config_path is not None else "configs/train.yaml"
    )

    config = load_transformer_config(config_path, args_config, args.on_unprocessed)

    column_types = {
        col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
        for col in config.column_types
    }

    data_train = read_data(config.training_data_path, config.read_format)
    if config.selected_columns is not None:
        data_train = subset_to_selected_columns(data_train, config.selected_columns)
    X_train, y_train = numpy_to_pytorch(
        data_train,
        column_types,
        config.target_column,
        config.seq_length,
        config.training_spec.device,
        to_device=True,
    )
    del data_train

    data_valid = read_data(config.validation_data_path, config.read_format)

    if config.selected_columns is not None:
        data_valid = subset_to_selected_columns(data_valid, config.selected_columns)

    X_valid, y_valid = numpy_to_pytorch(
        data_valid,
        column_types,
        config.target_column,
        config.seq_length,
        config.training_spec.device,
        to_device=True,
    )
    del data_valid

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = torch.compile(TransformerModel(config).to(config.training_spec.device))

    model.train_model(X_train, y_train, X_valid, y_valid)


def load_inference_model(
    model_path, training_config_path, args_config, device, infer_with_dropout
):
    training_config = load_transformer_config(
        training_config_path, args_config, args_config["on_unprocessed"]
    )

    with torch.no_grad():
      
        model = TransformerModel(training_config)
        model.log_file.write(f"Loading model weights from {model_path}")
        model_state = torch.load(model_path)
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


def infer_with_model(model, x, device):

    outs = np.concatenate(
        [
            model({col: torch.from_numpy(x_).to(device) for col, x_ in x_sub.items()})
            .cpu()
            .detach()
            .numpy()
            for x_sub in x
        ],
        axis=0,
    )

    return outs
