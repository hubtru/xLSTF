import torch
from torch.utils.data import DataLoader

from .. import BaseModel
from ..normalization import SIN
from .DFAN import DFAN


class SINFAN(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        train_dl: DataLoader,
        num_layers: int = 2,
        p_ratio: float = 0.25,
        use_p_bias: bool = True,
        gated: bool = False,
        **kwargs,
    ) -> None:
        super(SINFAN, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )

        self.norm = SIN(
            self.input_sequence_length,
            self.output_sequence_length,
            self.num_features,
            train_dl=train_dl,
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

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if y is not None:
            self.norm.train_module(x, y)

        x, theta_x = self.norm.forward(x, mode="norm")
        output = self.backbone.forward(x)
        output = self.norm.forward(output, mode="denorm", theta_x=theta_x)
        return output
