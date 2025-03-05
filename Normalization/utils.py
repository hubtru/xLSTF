import logging
import pathlib
from typing import Callable, Literal, Optional, Tuple, Type

import numpy as np
import torch
from prettytable import PrettyTable
from torch import nn

type ActivationFn = Literal["relu", "leaky_relu", "gelu", "none"]


def get_activation_fn(act_str: ActivationFn | None) -> Type[nn.Module]:
    match act_str:
        case "relu":
            return nn.ReLU
        case "leaky_relu":
            return nn.LeakyReLU
        case "gelu":
            return nn.GELU
        case "sish":
            return nn.SiLU
        case "none":
            return nn.Identity
        case None:
            return nn.Identity


class EarlyStopping(object):
    def __init__(
        self, patience: int = 7, verbose: bool = False, delta: float = 0.0
    ) -> None:
        self.patience = patience
        self.verbose = verbose

        self.counter = 0
        self.early_stop = False
        self.best_score = None
        self.best_valid_loss = float("inf")
        self.delta = delta

    def __call__(
        self, score: float, model: nn.Module, checkpoints_dir: pathlib.Path
    ) -> None:
        score = -score
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(score, model, checkpoints_dir)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(score, model, checkpoints_dir)
            self.counter = 0

    def _save_checkpoint(
        self, score: float, model: nn.Module, checkpoints_dir: pathlib.Path
    ) -> None:
        if self.verbose:
            print(
                f"Validation Loss decreased ({-self.best_valid_loss:.6f} --> {-score:.6f}).  Saving checkpoint...)"
            )
        torch.save(model.state_dict(), checkpoints_dir / "checkpoint.pth")
        self.best_valid_loss = score


def get_device() -> torch.device:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() and device == "cpu" else "cpu"
    return torch.device(device)


def flatten_dict(
    metric_dict: dict[str, list[float]],
    op: Literal["mean", "median", "sum", "min", "max"],
) -> dict[str, float]:
    """
    Flattens the lists contained in the metric_dict

    :param metric_dict: A dictionary of shape { metric_name: [values] }
    :param op: The flattening operation
    :return: The flattened dictionaries of shape { metric_name: reduced_value }
    """
    output = {}

    if op == "mean":
        op = np.mean
    elif op == "sum":
        op = np.sum
    elif op == "min":
        op = np.min
    elif op == "max":
        op = np.max
    elif op == "median":
        op = np.median
    else:
        raise NotImplementedError()

    for name in metric_dict.keys():
        output[name] = op(metric_dict[name]).item()
    return output


def log_results(results: dict[str, float], mode: Literal["Validation", "Test"]) -> None:
    logging.info(f"{mode} Results:")
    for metric_name, metric_value in results.items():
        logging.info(f"\t{metric_name}: {metric_value:.7f}")


def count_parameters(
    model: nn.Module, print_function: Optional[Callable] = None
) -> Tuple[int, float]:
    table = PrettyTable(["Modules", "Parameters"])
    table.align["Modules"] = "l"
    table.align["Parameters"] = "l"

    total_size = 0
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

        if torch.is_floating_point(parameter):
            total_size += param * torch.finfo(parameter.data.dtype).bits
        elif torch.is_complex(parameter):
            total_size += param * torch.finfo(parameter.data.dtype).bits * 2
        else:
            total_size += param * torch.iinfo(parameter.data.dtype).bits

    total_size = total_size / 8 / 1024 / 1024
    if print_function is not None:
        print_function(table)
        print_function(f"Total Trainable Params: {total_params} ({total_size:.2f} MB)")
    return total_params, total_size


def get_model_str(model: str, dataset: str, size: Tuple[int, int, int]) -> str:
    dataset_name = dataset.split(".")[0]
    if "ThreeColumnDataset" in dataset_name.split("/"):
        dataset_name = dataset_name.split("/")[-1]

    return f"{dataset_name}_{model}_{size[0]}_{size[-1]}_EXP"
