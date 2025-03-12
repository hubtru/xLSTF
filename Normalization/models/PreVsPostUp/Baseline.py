import torch

from Normalization.models import BaseModel
from Normalization.models.linear import NLinear
from Normalization.models.normalization import RevIN


class Baseline(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        individual: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            num_features=num_features,
        )

        self.norm = RevIN(num_features=num_features)
        self.backbone = NLinear(
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
        out = self.backbone(x)
        out = self.norm.forward(out, mode="denorm")
        return out
