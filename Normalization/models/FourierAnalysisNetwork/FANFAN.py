from typing import Tuple

import torch

from .. import BaseModel
from ..normalization import FAN as FANNorm
from .DFAN import DFAN


class FANFAN(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        num_layers: int = 2,
        p_ratio: float = 0.25,
        use_p_bias: bool = True,
        gated: bool = False,
        top_k_frequencies: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            num_features=num_features,
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

        self.top_k_frequencies = top_k_frequencies
        self.norm = FANNorm(
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            num_features=num_features,
            freq_topk=self.top_k_frequencies,
        )

    def params(self) -> dict:
        return {
            "norm": {"type": "FANNorm", "params": self.norm.params()},
            "backbone": {"type": "FAN", "params": self.backbone.params()},
        }

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_norm, pt1 = self.norm.normalize(x)
        out = self.backbone(x_norm)
        out_denorm, pt2 = self.norm.denormalize(out, pt1)
        return out_denorm, (pt1, pt2)
