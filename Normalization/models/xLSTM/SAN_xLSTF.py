from typing import Literal

import torch
from torch.utils.data import DataLoader

from ..BaseModel import BaseModel
from ..normalization import SAN
from .xLSTF import xLSTF


class SAN_xLSTF(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        dropout_prob: float = 0.0,
        **kwargs,
    ) -> None:
        super(SAN_xLSTF, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )

        self.norm = SAN(
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            num_features=num_features,
        )
        self.backbone = xLSTF(
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

    def pretrain_model(
        self,
        train_dl: DataLoader,
        features: Literal["M", "S", "MS"] = "M",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.norm.train_module(train_dl, features, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, input_length, num_features)
        x, stats = self.norm.normalize(x)
        output = self.backbone(x)
        output = self.norm.denormalize(output, stats)

        # output: (bs, output_length, num_features)
        return output
