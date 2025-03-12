from typing import Tuple

import torch

import Normalization

from ..BaseModel import BaseModel
from ..normalization import FAN


class FAN_xLSTF(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        dropout_prob: float = 0.0,
        **kwargs,
    ) -> None:
        super(FAN_xLSTF, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )

        self.norm = FAN(
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            num_features=num_features,
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

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (bs, input_length, num_features)
        x, pt1 = self.norm.normalize(x)
        output = self.backbone(x)
        output, pt2 = self.norm.denormalize(output, pt1)
        # output: (bs, output_length, num_features)
        return output, (pt1, pt2)
