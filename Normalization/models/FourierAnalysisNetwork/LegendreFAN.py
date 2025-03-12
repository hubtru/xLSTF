import torch
from torch import nn

from Normalization.models import BaseModel
from Normalization.models.normalization import LegendreProjectionUnit, RevIN

from .FAN import FAN


class LegendreFAN(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        use_RevIN: bool = False,
        num_polynomials: int = 128,
        share_weights: bool = True,
        num_layers: int = 2,
        p_ratio: float = 0.25,
        use_p_bias: bool = True,
        gated: bool = False,
        **kwargs,
    ) -> None:
        super(LegendreFAN, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )

        self.backbone = (
            nn.ModuleList(
                [
                    FAN(
                        input_sequence_length,
                        output_sequence_length,
                        num_polynomials,
                        num_layers,
                        p_ratio,
                        use_p_bias,
                        gated=gated,
                    )
                    for _ in range(num_features)
                ]
            )
            if not share_weights
            else FAN(
                input_sequence_length,
                output_sequence_length,
                num_polynomials,
                num_layers,
                p_ratio,
                use_p_bias,
                gated=gated,
            )
        )
        self.use_revin = use_RevIN
        self.norm = RevIN(num_features=num_features) if use_RevIN else None
        self.lpu = LegendreProjectionUnit(
            N=num_polynomials, dt=1 / self.output_sequence_length
        )
        self.share_weights = share_weights

    def params(self) -> dict:
        return {
            "norm": {
                "type": str(self.norm.__class__.__name__),
                "params": self.norm.params(),
            }
            if self.use_revin
            else None,
            "legendre_projection": {
                "type": str(self.lpu.__class__.__name__),
                "params": self.lpu.params(),
            },
            "backbone": {
                "type": str(self.backbone.__class__.__name__),
                "params": self.backbone.params(),
                "share_weights": self.share_weights,
            },
        }

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x_norm = self.norm.forward(x, mode="norm") if self.norm is not None else x
        x_c = self.lpu.forward(x_norm).transpose(-1, -2)

        inputs_c = [inp.squeeze(1) for inp in x_c.chunk(x_c.shape[1], dim=1)]

        print(inputs_c[0].shape, len(inputs_c))

        if self.share_weights:
            outputs_c = [self.backbone(inputs_c[i]) for i in range(len(inputs_c))]
        else:
            outputs_c = [model(inputs_c[i]) for i, model in enumerate(self.backbone)]

        outputs_c = torch.stack(outputs_c, dim=1).transpose(-1, -2)
        outputs = self.lpu.reconstruct(
            outputs_c, self.input_sequence_length, self.output_sequence_length
        )
        outputs = (
            self.norm.forward(outputs, mode="denorm")
            if self.norm is not None
            else outputs
        )
        return outputs
