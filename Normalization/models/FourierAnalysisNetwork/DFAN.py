import torch
from torch import nn

from xLSTF.models import BaseModel
from xLSTF.models.utils import SeriesDecomposition

from .FAN import FAN


class DFAN(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        decomposition_kernel_size: int = 25,
        num_layers: int = 2,
        p_ratio: float = 0.25,
        use_p_bias: bool = True,
        gated: bool = False,
        **kwargs,
    ) -> None:
        super(DFAN, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )

        self.decomposition_kernel_size = decomposition_kernel_size
        self.decomp = SeriesDecomposition(kernel_size=decomposition_kernel_size)
        self.backbones = nn.ModuleDict(
            {
                "trend": FAN(
                    input_sequence_length,
                    output_sequence_length,
                    num_features,
                    num_layers,
                    p_ratio,
                    use_p_bias,
                    gated,
                ),
                "seasonal": FAN(
                    input_sequence_length,
                    output_sequence_length,
                    num_features,
                    num_layers,
                    p_ratio,
                    use_p_bias,
                    gated,
                ),
            }
        )

    def params(self) -> dict:
        return {
            "norm": {
                "type": "Trend-Residual-Decomposition",
                "params": {"decomposition_kernel_size": self.decomposition_kernel_size},
            },
            "trend_projection": {
                "type": str(self.backbones["trend"].__class__.__name__),
                "params": self.backbones["trend"].params(),
            },
            "seasonal_projection": {
                "type": str(self.backbones["seasonal"].__class__.__name__),
                "params": self.backbones["seasonal"].params(),
            },
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend, seasonal = self.decomp(x)
        trend = self.backbones["trend"](trend)
        seasonal = self.backbones["seasonal"](seasonal)
        return trend + seasonal
