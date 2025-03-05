from typing import Protocol

import torch


class Metric(Protocol):
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        pass


class MeanAbsoluteError(Metric):
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        return torch.mean(torch.abs(y_pred - y_true)).item()


class MeanSquaredError(Metric):
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        return torch.mean((y_pred - y_true) ** 2).item()


class RootMeanSquaredError(Metric):
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()


AVAILABLE_METRICS: dict[str, Metric] = {
    "MAE": MeanAbsoluteError(),
    "MSE": MeanSquaredError(),
    "RMSE": RootMeanSquaredError(),
}
