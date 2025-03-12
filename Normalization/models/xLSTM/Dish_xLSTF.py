from typing import Literal, Optional

import torch
from torch import nn

import Normalization

from ..BaseModel import BaseModel
from ..normalization import DishTS


class Dish_xLSTF(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        dropout_prob: float = 0.0,
        dish_alpha: float = 0.0,
        dish_init: Literal["standard", "avg", "uniform"] = "standard",
        dish_activation: Optional[nn.Module] = nn.GELU,
        **kwargs,
    ) -> None:
        super(Dish_xLSTF, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )

        self.norm = DishTS(
            input_sequence_length, num_features, dish_alpha, dish_init, dish_activation
        )
        self.backbone = Normalization.models.xLSTM.xLSTF(
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            num_features=num_features,
            use_RevIN=False,
            dropout_prob=dropout_prob,
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
