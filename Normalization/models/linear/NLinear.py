import torch

from .. import BaseModel
from .Linear import Linear


class NLinear(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        individual: bool = False,
        **kwargs,
    ) -> None:
        super(NLinear, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )
        self.backbone = Linear(
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            num_features=num_features,
            individual=individual,
        )

    def params(self) -> dict:
        return {
            "norm": {"type": "N-Normalization", "params": None},
            "backbone": {
                "type": str(self.backbone.__class__.__name__),
                "params": self.backbone.params(),
            },
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.backbone.forward(x)
        x = x + seq_last
        return x
