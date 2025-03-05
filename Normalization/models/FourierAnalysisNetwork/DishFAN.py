from typing import Literal

import torch
from torch import nn

from .. import BaseModel
from ..normalization import DishTS
from .DFAN import DFAN


class DishFAN(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        num_layers: int = 2,
        p_ratio: float = 0.25,
        use_p_bias: bool = True,
        gated: bool = False,
        dish_alpha: float = 0.0,
        dish_init: Literal["standard", "avg", "uniform"] = "standard",
        **kwargs,
    ) -> None:
        super(DishFAN, self).__init__(
            input_sequence_length,
            output_sequence_length,
            num_features,
        )

        self.norm = DishTS(
            input_sequence_length,
            num_features,
            dish_alpha,
            dish_init,
            dish_activation=nn.LeakyReLU,
        )
        self.backbone = DFAN(
            input_sequence_length,
            output_sequence_length,
            num_features,
            num_layers=num_layers,
            p_ratio=p_ratio,
            use_p_bias=use_p_bias,
            gated=gated,
        )

    def params(self) -> dict:
        return {
            "norm": {
                "type": str(self.norm.__class__.__name__),
                "params": self.norm.params(),
            },
            "backbone": {
                "type": str(self.backbone.__class__.__name__),
                "params": self.norm.params(),
            },
        }

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, phih = self.norm.forward(x, mode="norm")
        x = self.backbone.forward(x)
        x = self.norm.forward(x, mode="denorm")
        return x, phih
