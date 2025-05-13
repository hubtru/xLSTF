import collections
import logging
import time
from pathlib import Path
from typing import Callable, Literal, Tuple, Type

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from xLSTF.configs import ModelConfiguration, TrainingConfiguration
from xLSTF.metrics import AVAILABLE_METRICS
from xLSTF.models import BaseModel
from xLSTF.models.normalization import FAN, SAN, SIN, pre_train_stat_model
from xLSTF.utils import EarlyStopping, flatten_dict


def custom_loss(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return 0.5 * F.mse_loss(pred, true) + 0.5 * F.l1_loss(pred, true)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    validation_loader: DataLoader,
    prediction_length: int,
    metrics_to_compute: dict[str, Callable],
    features: Literal["MS", "S", "M"] = "MS",
    device: torch.device = torch.device("cuda:0"),
) -> dict[str, float]:
    """
    Validates the model. Is also used for testing
    :param model: The model to validate/test
    :param validation_loader: A dataloader with validation/testing data
    :param prediction_length: The length of the prediction sequence (forecasting horizon)
    :param metrics_to_compute: A dictionary of metrics to compute during validation
    :param features:
    :param device: The device on which the computation should be performed
    :return: A dictionary of shape { metric_name: mean(metric_values) }
    """
    model.eval()
    feature_dim = -1 if features == "MS" else 0

    out_dict = collections.defaultdict(list)
    for x, y in validation_loader:
        x = x.float().to(device)
        y = y.float().to(device)
        out = model.forward(x)
        if isinstance(out, tuple):
            out = out[0]

        y = y[:, -prediction_length:, feature_dim:]
        out = out[:, -prediction_length:, feature_dim:]

        for name, func in metrics_to_compute.items():
            out_dict[name].append(func(y, out))

    return flatten_dict(out_dict, op="mean")


def train_epoch(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    prediction_length: int,
    features: Literal["MS", "M", "S"] = "MS",
    device: torch.device = torch.device("cuda:0"),
) -> float:
    """
    Trains the model for one epoch
    :param model: The model to train
    :param loss_fn: The loss function to use
    :param optimizer: The optimizer to use
    :param train_loader: A dataloader for the training data
    :param prediction_length: The length of the prediction sequence (forecasting horizon)
    :param features:
    :param device: The device on which the computation should be performed
    :return: The average loss of the model
    """
    model.train()
    feature_dim = -1 if features == "MS" else 0

    losses = []
    for x, y in train_loader:
        x = x.float().to(device)
        y = y.float().to(device)

        out = model.forward(x)
        if isinstance(out, tuple):
            out, stats = out

        y = y[:, -prediction_length:, feature_dim:]
        out = out[:, -prediction_length:, feature_dim:]
        if hasattr(model, "norm") and isinstance(model.norm, FAN):
            loss = model.norm.normalization_loss(y, stats, loss_fn) + loss_fn(out, y)
        else:
            loss = loss_fn(out, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(losses).item()


def train_model(
    model: Type[BaseModel],
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    learning_rate: float,
    epochs: int,
    input_sequence_length: int,
    output_sequence_length: int,
    num_features: int,
    checkpoints_dir: Path,
    features: Literal["MS", "S", "M"] = "MS",
    early_stopping_patience: int = 5,
    metrics_to_compute: dict[str, Callable] | None = None,
    device: torch.device = torch.device("cuda:0"),
) -> tuple[int, float, dict[str, float], dict[str, float], dict[str, float]]:
    if metrics_to_compute is None:
        metrics_to_compute = AVAILABLE_METRICS

    model = model(
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
        num_features=num_features,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    early_stopping = EarlyStopping(patience=early_stopping_patience)
    validation_results, test_results = (
        collections.defaultdict(list),
        collections.defaultdict(list),
    )
    timings = {"training": [], "validation": [], "testing": []}
    training_losses = []

    if hasattr(model, "norm") and isinstance(model.norm, SAN):
        san_optimizer = torch.optim.AdamW(model.norm.parameters(), lr=learning_rate)
        pre_train_stat_model(
            model.norm,
            san_optimizer,
            train_loader,
            model.norm.patch_length,
            output_sequence_length,
            epochs=5,
            features=features,
            verbose=True,
            device=device,
        )
        model.norm.statistics_prediction_model_is_trained = True
        model.norm.model.requires_grad_(False)
    if hasattr(model, "norm") and isinstance(model.norm, SIN):
        logging.info("Training the SIN module first...")
        x_full, y_full = train_loader.dataset.data_x, train_loader.dataset.data_y
        x_full, y_full = (
            torch.from_numpy(x_full).float().to(device),
            torch.from_numpy(y_full).float().to(device),
        )
        model.norm.train_module(x_full, y_full)

    max_epoch = -1
    for epoch in range(epochs):
        train_start_time = time.time()
        train_loss = train_epoch(
            model,
            loss_fn,
            optimizer,
            train_loader,
            output_sequence_length,
            features,
            device=device,
        )
        training_time = time.time() - train_start_time
        timings["training"].append(training_time)
        training_losses.append(train_loss)

        validation_start_time = time.time()
        valid_outputs = validate_epoch(
            model,
            valid_loader,
            output_sequence_length,
            metrics_to_compute,
            features,
            device=device,
        )
        validation_time = time.time() - validation_start_time
        timings["validation"].append(validation_time)

        test_start_time = time.time()
        test_outputs = validate_epoch(
            model,
            test_loader,
            output_sequence_length,
            metrics_to_compute,
            features,
            device=device,
        )
        test_time = time.time() - test_start_time
        timings["testing"].append(test_time)

        for metric_name in metrics_to_compute.keys():
            validation_results[metric_name].append(valid_outputs[metric_name])
            test_results[metric_name].append(test_outputs[metric_name])

        logging.info(
            "Epoch: {0:02d} | Train Loss (MAE): {1:.7f} | Validation Loss (MAE/MSE): {2:.7f}/{3:.7f} | Test Loss (MAE/MSE): {4:.7f}/{5:.7f}".format(
                epoch + 1,
                train_loss,
                valid_outputs["MAE"],
                valid_outputs["MSE"],
                test_outputs["MAE"],
                test_outputs["MSE"],
            )
        )
        max_epoch = epoch + 1
        early_stopping(valid_outputs["MAE"], model, checkpoints_dir)
        if early_stopping.early_stop:
            logging.info("Early Stopping...")
            break

    return (
        max_epoch,
        np.min(training_losses).item(),
        flatten_dict(validation_results, op="min"),
        flatten_dict(test_results, op="min"),
        flatten_dict(timings, op="mean"),
    )


def train(
    model_cfg: ModelConfiguration,
    train_cfg: TrainingConfiguration,
    datasets: Tuple[Dataset, Dataset, Dataset],
    checkpoints_dir: Path,
    num_experiments: int = 1,
) -> tuple[
    list[int] | int, float, dict[str, float], dict[str, float], dict[str, float]
]:
    train_loader = DataLoader(
        datasets[0],
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=4,
    )
    valid_loader = DataLoader(
        datasets[1],
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        datasets[2],
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=4,
    )

    final_valid_results, final_test_results = (
        collections.defaultdict(list),
        collections.defaultdict(list),
    )
    max_epochs = []
    timings = {"training": [], "validation": [], "testing": []}
    training_losses = []
    for _ in range(num_experiments):
        max_epoch, avg_loss, valid_results, test_results, exp_timings = train_model(
            model_cfg.model,
            train_loader,
            valid_loader,
            test_loader,
            train_cfg.loss_fn,
            learning_rate=train_cfg.learning_rate,
            epochs=train_cfg.max_epochs,
            input_sequence_length=model_cfg.input_sequence_length,
            output_sequence_length=model_cfg.output_sequence_length,
            num_features=model_cfg.num_features,
            checkpoints_dir=checkpoints_dir,
            features=train_cfg.features,
            device=train_cfg.device,
        )
        assert valid_results.keys() == test_results.keys()
        for metric_name in test_results.keys():
            final_valid_results[metric_name].append(valid_results[metric_name])
            final_test_results[metric_name].append(test_results[metric_name])

        assert exp_timings.keys() == timings.keys()
        for key in exp_timings.keys():
            timings[key].append(exp_timings[key])

        training_losses.append(avg_loss)
        max_epochs.append(max_epoch)

    return (
        max_epochs if len(max_epochs) > 1 else max_epochs[0],
        np.mean(training_losses).item(),
        flatten_dict(final_valid_results, op="mean"),
        flatten_dict(final_test_results, op="mean"),
        flatten_dict(timings, op="mean"),
    )
