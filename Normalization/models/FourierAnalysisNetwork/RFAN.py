import torch

from Normalization.models import BaseModel

from ..normalization import RevIN
from .DFAN import DFAN


class RFAN(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        affine: bool = True,
        num_layers: int = 2,
        p_ratio: float = 0.25,
        use_p_bias: bool = True,
        gated: bool = False,
        **kwargs,
    ) -> None:
        super(RFAN, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )
        self.norm = RevIN(num_features, affine=affine)
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
                "params": self.backbone.params(),
            },
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm.forward(x, mode="norm")
        x = self.backbone(x)
        x = self.norm.forward(x, mode="denorm")
        return x
