from typing import Literal, Tuple

import torch

from .. import BaseModel
from ..normalization import FAN
from . import DLinear, NLinear


class FANLinear(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        decomposition_kernel_size: int = 25,
        individual: bool = False,
        top_k_frequencies: int = 20,
        backbone: Literal["DLinear", "NLinear"] = "DLinear",
        **kwargs,
    ) -> None:
        super().__init__(
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            num_features=num_features,
        )

        self.backbone = (
            DLinear(
                input_sequence_length=input_sequence_length,
                output_sequence_length=output_sequence_length,
                num_features=num_features,
                decomposition_kernel_size=decomposition_kernel_size,
                individual=individual,
            )
            if backbone == "DLinear"
            else NLinear(
                input_sequence_length=input_sequence_length,
                output_sequence_length=output_sequence_length,
                num_features=num_features,
                individual=individual,
            )
        )

        self.top_k_frequencies = top_k_frequencies
        self.norm = FAN(
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            num_features=num_features,
            freq_topk=self.top_k_frequencies,
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
        x_norm, pt1 = self.norm.normalize(x)
        out = self.backbone(x_norm)
        out_denorm, pt2 = self.norm.denormalize(out, pt1)
        return out_denorm, (pt1, pt2)
