import torch

from .. import BaseModel
from ..normalization import RevIN
from . import DLinear


class RLinear(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        individual: bool = False,
        affine: bool = True,
        **kwargs,
    ) -> None:
        super(RLinear, self).__init__(
            input_sequence_length,
            output_sequence_length,
            num_features,
        )

        self.norm = RevIN(num_features=num_features, affine=affine)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm.forward(x, mode="norm")
        x = self.backbone.forward(x)
        x = self.norm.forward(x, mode="denorm")
        return x
