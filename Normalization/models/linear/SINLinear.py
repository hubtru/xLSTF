from typing import Literal

import torch
from torch.utils.data import DataLoader

from .. import BaseModel
from ..normalization import SIN
from . import DLinear, NLinear


class SINLinear(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        train_dl: DataLoader,
        backbone: Literal["DLinear", "NLinear"] = "DLinear",
        decomposition_kernel_size: int = 25,
        individual: bool = False,
        **kwargs,
    ) -> None:
        super(SINLinear, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )

        self.norm = SIN(
            self.input_sequence_length,
            self.output_sequence_length,
            self.num_features,
            train_dl=train_dl,
        )

        self.backbone = (
            DLinear(
                self.input_sequence_length,
                self.output_sequence_length,
                self.num_features,
                decomposition_kernel_size=decomposition_kernel_size,
                individual=individual,
            )
            if backbone == "DLinear"
            else NLinear(
                self.input_sequence_length,
                self.output_sequence_length,
                self.num_features,
                individual=individual,
            )
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, theta_x = self.norm.normalize(x)
        output = self.backbone.forward(x)
        output = self.norm.denormalize(output, theta_x)
        return output
