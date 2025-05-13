from typing import Literal, Optional

import torch
from torch import nn

from .. import BaseModel
from ..normalization import DishTS
from .DLinear import DLinear


class DishLinear(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        individual: bool = False,
        dish_alpha: float = 0.0,
        dish_init: Literal["standard", "avg", "uniform"] = "standard",
        dish_activation: Optional[nn.Module] = nn.GELU,
        **kwargs,
    ) -> None:
        super(DishLinear, self).__init__(
            input_sequence_length,
            output_sequence_length,
            num_features,
        )

        self.norm = DishTS(
            input_sequence_length, num_features, dish_alpha, dish_init, dish_activation
        )
        self.backbone = DLinear(
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            num_features=num_features,
            individual=individual,
        )

    def params(self) -> dict:
        return {
            "norm": {
                "type": str(self.norm.__class__.__name__),
                "params": self.norm.params(),
            },
            "backbone": {
                "type": str(self.backbone.__class__.__name__),
                "params": self.backbone.params(),
            },
        }

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, phih = self.norm.forward(x, mode="norm")
        x = self.backbone.forward(x)
        x = self.norm.forward(x, mode="denorm")
        return x, phih
