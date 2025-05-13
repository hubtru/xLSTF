from typing import Literal

import torch
from torch.utils.data import DataLoader

from .. import BaseModel
from ..normalization import SAN
from . import NLinear
from .DLinear import DLinear


class SANLinear(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        decomposition_kernel_size: int = 25,
        backbone: Literal["DLinear", "NLinear"] = "DLinear",
        individual: bool = False,
        **kwargs,
    ) -> None:
        super(SANLinear, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )

        self.norm = SAN(
            input_sequence_length=self.input_sequence_length,
            output_sequence_length=self.output_sequence_length,
            num_features=self.num_features,
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

    def pretrain_model(
        self,
        train_dl: DataLoader,
        features: Literal["M", "S", "MS"] = "M",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.norm.train_module(train_dl, features, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, norm_output = self.norm.forward(x, None, mode="norm")
        output = self.backbone.forward(x)
        output = self.norm.forward(output, norm_output, mode="denorm")
        return output
