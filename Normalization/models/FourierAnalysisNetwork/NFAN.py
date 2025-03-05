import torch

from xLSTF.models import BaseModel

from . import FAN


class NFAN(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        num_layers: int = 2,
        p_ratio: float = 0.25,
        use_p_bias: bool = True,
        gated: bool = False,
        **kwargs,
    ) -> None:
        super(NFAN, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )
        self.backbone = FAN(
            input_sequence_length,
            output_sequence_length,
            num_features,
            num_layers,
            p_ratio,
            activation="gelu",
            use_p_bias=use_p_bias,
            gated=gated,
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
        x = self.backbone(x)
        x = x + seq_last
        return x
