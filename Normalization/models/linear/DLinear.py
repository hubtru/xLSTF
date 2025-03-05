import torch
from torch import nn

from ..BaseModel import BaseModel
from ..utils import SeriesDecomposition


class DLinear(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        decomposition_kernel_size: int = 25,
        individual: bool = False,
        **kwargs,
    ) -> None:
        super(DLinear, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )

        self.individual = individual
        kernel_size: int = decomposition_kernel_size
        self.series_decomposition = SeriesDecomposition(kernel_size=kernel_size)

        if self.individual:
            self.weights_trend = nn.ModuleList()
            self.weights_residual = nn.ModuleList()

            for i in range(self.num_features):
                self.weights_trend.append(
                    nn.Linear(input_sequence_length, output_sequence_length)
                )
                self.weights_residual.append(
                    nn.Linear(input_sequence_length, output_sequence_length)
                )
        else:
            self.weights_trend = nn.Linear(
                input_sequence_length, output_sequence_length
            )
            self.weights_residual = nn.Linear(
                input_sequence_length, output_sequence_length
            )

        self.decomposition_kernel_size = decomposition_kernel_size

    def params(self) -> dict:
        return {
            "norm": {
                "type": "Trend-Residual-Decomposition",
                "params": {"decomposition_kernel_size": self.decomposition_kernel_size},
            },
            "trend_projection": {
                "type": "Linear",
                "params": {"individual": self.individual},
            },
            "seasonal_projection": {
                "type": "Linear",
                "params": {"individual": self.individual},
            },
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.series_decomposition(x)
        seasonal, trend = seasonal.permute(0, 2, 1), trend.permute(0, 2, 1)

        if self.individual:
            trend_output = torch.zeros(
                (x.shape[0], self.num_features, self.output_sequence_length),
                dtype=x.dtype,
                device=x.device,
            )
            seasonal_output = torch.zeros(
                (x.shape[0], self.num_features, self.output_sequence_length),
                dtype=x.dtype,
                device=x.device,
            )

            for i in range(self.num_features):
                trend_output[:, i, :] = self.weights_trend[i](trend[:, i, :])
                seasonal_output[:, i, :] = self.weights_residual[i](seasonal[:, i, :])
        else:
            trend_output = self.weights_trend(trend)
            seasonal_output = self.weights_residual(seasonal)

        x = (trend_output + seasonal_output).permute(0, 2, 1)
        return x
