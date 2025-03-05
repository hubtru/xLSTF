from typing import Optional

import torch
from torch import nn

from xLSTF.models import BaseModel
from xLSTF.models.linear import NLinear
from xLSTF.models.normalization import RevIN
from xLSTF.models.PreVsPostUp.utils import SkipConnectionType
from xLSTF.utils import ActivationFn, get_activation_fn


class Right(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        individual: bool = False,
        hidden_factor: float = 1.0,
        activation: Optional[ActivationFn] = None,
        skip_connection: Optional[SkipConnectionType] = None,
    ) -> None:
        super(Right, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )
        self.norm = RevIN(num_features=num_features, affine=True)

        self.hidden_channels: int = int(num_features * hidden_factor)
        self.up_proj = nn.Linear(num_features, self.hidden_channels)
        self.down_proj = nn.Linear(self.hidden_channels, num_features)
        self.activation_fn = get_activation_fn(activation)()
        self.skip_connection = skip_connection

        self.forecasting_model = NLinear(
            input_sequence_length, output_sequence_length, num_features, individual
        )

    def params(self) -> dict:
        return {
            "norm": {
                "type": str(self.norm.__class__.__name__),
                "params": self.norm.params(),
            },
            "backbone": {
                "type": str(self.forecasting_model.__class__.__name__),
                "params": self.forecasting_model.params(),
            },
            "channel_proj": {
                "type": "Right",
                "params": {
                    "hidden_dim": self.hidden_channels,
                    "activation_fn": str(self.activation_fn.__class__.__name__),
                    "skip_connection": str(self.skip_connection),
                },
            },
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm.forward(x, mode="norm")
        out = self.forecasting_model(x)

        if self.skip_connection == SkipConnectionType.SKIP:
            out_residual = out

        out = self.up_proj(out)
        out = self.activation_fn(out)
        out = self.down_proj(out)

        if self.skip_connection == SkipConnectionType.SKIP:
            out = out + out_residual

        out = self.norm.forward(out, mode="denorm")
        return out


def main() -> None:
    x = torch.randn((32, 255, 10), dtype=torch.float32)
    model = Right(255, 128, 10, False, 32, skip_connection=SkipConnectionType.SKIP)
    print(model(x).shape)


if __name__ == "__main__":
    main()
