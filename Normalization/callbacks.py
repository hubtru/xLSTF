from typing import Literal, Tuple

import numpy as np
from lightning import pytorch as L

from Normalization.utils import count_parameters


class ParameterCounterCallback(L.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.param_count, self.total_size = None, None

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        count, total_size = count_parameters(pl_module.model, print_function=print)
        self.param_count, self.total_size = count, total_size

    def get_parameter_stats(self) -> Tuple[int, float]:
        return self.param_count, self.total_size


class LossCallback(L.Callback):
    def __init__(self, reduce_op: Literal["mean", "min", "last"] = "last") -> None:
        super().__init__()
        self.reduce_op = reduce_op
        self.losses = {"train": [], "valid": []}

    def _get_single(self, key: str) -> float:
        match self.reduce_op:
            case "mean":
                return np.mean(self.losses[key]).item()
            case "min":
                return np.min(self.losses[key]).item()
            case "last":
                return self.losses[key][-1].item()
            case _:
                raise NotImplementedError()

    def get_loss(
        self, key: Literal["all", "train", "val"]
    ) -> float | Tuple[float, float]:
        match key:
            case "all":
                return self._get_single("train"), self._get_single("valid")
            case "train":
                return self._get_single("train")
            case "val":
                return self._get_single("valid")
            case _:
                raise NotImplementedError()

    def on_train_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        self.losses["train"].append(trainer.callback_metrics["train/loss"])

    def on_validation_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        self.losses["valid"].append(trainer.callback_metrics["val/MAE"])
