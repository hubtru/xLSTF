from typing import Optional

import torch
from torch import nn

from xLSTF.models import BaseModel
from xLSTF.models.normalization import LegendreProjectionUnit, RevIN


class LegendreLinear(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        use_RevIN: bool = False,
        num_polynomials: int = 256,
        individual: bool = False,
        low_rank_approximation: Optional[int] = 16,
        **kwargs,
    ) -> None:
        super(LegendreLinear, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )

        num_polynomials = num_polynomials
        self.lpu = LegendreProjectionUnit(
            N=num_polynomials, dt=1 / self.output_sequence_length
        )

        self.individual = individual
        self.backbone = (
            nn.Linear(input_sequence_length, 1)
            if not self.individual
            else nn.ModuleList(
                [nn.Linear(input_sequence_length, 1) for _ in range(self.num_features)]
            )
        )

        self.low_rank_approximation = None
        if low_rank_approximation is not None:
            self.low_rank_approximation = low_rank_approximation
            self.down_proj = nn.Linear(num_polynomials, low_rank_approximation)
            self.up_proj = nn.Linear(low_rank_approximation, num_polynomials)
        if use_RevIN:
            self.norm = RevIN(num_features, affine=True)

        self.projection_required = False

    def params(self) -> dict:
        return {
            "norm": {"type": "RevIN", "params": self.norm.params()}
            if hasattr(self, "norm")
            else None,
            "legendre_projection": {
                "type": "LegendreProjection",
                "params": self.lpu.params(),
            },
            "backbone": {
                "type": "Linear",
                "params": {
                    "individual": self.individual,
                },
            },
            "low_rank_approximation": {"hidden_channels": self.low_rank_approximation}
            if self.low_rank_approximation is not None
            else None,
        }

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # x: [bs, seq_len, num_variates]

        if hasattr(self, "norm"):
            x = self.norm.forward(x, mode="norm")  # -> [bs, seq_len, num_variates]

        x_c = self.lpu.forward(x)  # -> [bs, num_variates, num_polynomials, seq_len]

        if hasattr(self, "down_proj"):
            x_c = self.down_proj(
                x_c.transpose(-1, -2)
            )  # -> [bs, num_variates, seq_len, low_rank_approximation]
            x_c = x_c.transpose(
                -1, -2
            )  # -> [bs, num_variates, low_rank_approximation, seq_len]

        if not self.individual:
            out_c = self.backbone(x_c)
        else:
            assert len(self.backbone) == x_c.shape[1]
            out_cs = [
                model_fn(x_c[:, i, :, :]) for i, model_fn in enumerate(self.backbone)
            ]
            out_c = torch.stack(out_cs, dim=1)

        if hasattr(self, "up_proj"):
            out_c = self.up_proj(
                out_c.transpose(-1, -2)
            )  # -> [bs, num_variates, 1, num_polynomials]
            out_c = out_c.transpose(-1, -2)  # -> [bs, num_variates, num_polynomials, 1]

        out = (
            out_c.squeeze(-1)
            @ self.lpu.eval_matrix[-self.output_sequence_length :, :].T
        )  # -> [bs, num_variates, pred_len]
        out = out.transpose(-1, -2)  # -> [bs, pred_len, num_variates]

        if hasattr(self, "norm"):
            out = self.norm.forward(out, mode="denorm")  # [bs, pred_len, num_variates]

        return out
