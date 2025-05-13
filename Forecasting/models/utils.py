import json
from typing import Self

import torch
from torch import nn


def format_instance_str(cls: Self, params: dict) -> str:
    return f"{cls.__class__.__name__}({json.dumps(params)})"


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int, stride: int) -> None:
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front_padding = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        back_padding = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)

        x = torch.cat([front_padding, x, back_padding], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super(SeriesDecomposition, self).__init__()
        self.moving_average = MovingAverage(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_avg = self.moving_average(x)
        residual = x - moving_avg
        return residual, moving_avg
