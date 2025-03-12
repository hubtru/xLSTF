import torch
from torch import nn

from Normalization.models import BaseModel
from Normalization.utils import get_activation_fn


class FAN(BaseModel):
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
        super(FAN, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )

        activation_fn = get_activation_fn("gelu")
        self.layers = nn.ModuleList(
            [
                FANLayer(
                    input_sequence_length,
                    input_sequence_length,
                    p_ratio,
                    activation_fn,
                    use_p_bias,
                    gated,
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.output_proj = nn.Linear(input_sequence_length, output_sequence_length)

        (
            self.num_layers,
            self.p_ratio,
            self.activation_fn,
            self.use_p_bias,
            self.gated,
        ) = (num_layers, p_ratio, activation_fn, use_p_bias, gated)

    def params(self) -> dict:
        return {
            "num_layers": self.num_layers,
            "p_ratio": self.p_ratio,
            "activation_fn": str(self.activation_fn().__class__.__name__),
            "use_p_bias": self.use_p_bias,
            "gated": self.gated,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        out = self.output_proj(x.transpose(-1, -2)).transpose(-1, -2)
        return out


class FANLayer(nn.Module):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        p_ratio: float = 0.25,
        activation: nn.Module = nn.GELU,
        use_p_bias: bool = True,
        gated: bool = False,
    ) -> None:
        super(FANLayer, self).__init__()
        assert 0 < p_ratio < 0.5, "p_ratio must be in the range (0, 0.5)"

        p_output_dim = int(output_sequence_length * p_ratio)
        g_output_dim = output_sequence_length - p_output_dim * 2

        self.input_linear_p = nn.Linear(
            input_sequence_length, p_output_dim, bias=use_p_bias
        )
        self.input_linear_g = nn.Linear(input_sequence_length, g_output_dim)
        self.activation = activation()

        if gated:
            self.gate = nn.Parameter(
                torch.randn((1,), dtype=torch.float32), requires_grad=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [bs, input_sequence_length, num_features]
        x = x.permute(0, 2, 1)  # [bs, num_features, input_sequence_length]
        g = self.activation(self.input_linear_g(x))  # [bs, num_features, g_output_dim]
        p = self.input_linear_p(x)  # [bs, num_features, p_output_dim]

        if hasattr(self, "gate"):
            gate = torch.sigmoid(self.gate)
            output = torch.cat(
                [gate * torch.cos(p), gate * torch.sin(p), (1.0 - gate) * g], dim=-1
            )
        else:
            output = torch.cat(
                [torch.cos(p), torch.sin(p), g], dim=-1
            )  # [bs, num_features, output_sequence_length]
        output = output.permute(0, 2, 1)  # [bs, output_sequence_length, num_features]
        return output
