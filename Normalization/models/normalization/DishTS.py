from typing import Literal, Optional

import torch
from torch import nn


class DishTS(nn.Module):
    def __init__(
        self,
        input_sequence_length: int,
        num_features: int,
        alpha: float = 0.25,
        dish_init: Literal["standard", "avg", "uniform"] = "standard",
        dish_activation: Optional[nn.Module] = nn.LeakyReLU,
    ) -> None:
        super(DishTS, self).__init__()
        self.dish_init = dish_init
        match dish_init:
            case "standard":
                self.reduce_mlayer = nn.Parameter(
                    torch.randn(num_features, input_sequence_length, 2)
                    / input_sequence_length
                )
            case "avg":
                self.reduce_mlayer = nn.Parameter(
                    torch.ones(num_features, input_sequence_length, 2)
                    / input_sequence_length
                )
            case "uniform":
                self.reduce_mlayer = nn.Parameter(
                    torch.ones(num_features, input_sequence_length, 2)
                    / input_sequence_length
                    + torch.randn(num_features, input_sequence_length, 2)
                    / input_sequence_length
                )
        self.alpha = alpha
        self.gamma, self.beta = (
            nn.Parameter(torch.ones(num_features)),
            nn.Parameter(torch.zeros(num_features)),
        )
        self.activation = (
            dish_activation() if dish_activation is not None else nn.Identity()
        )

    def params(self) -> dict:
        return {
            "dish_weight_init": self.dish_init,
            "activation": self.activation.__class__.__name__,
        }

    def normalization_loss(self, y: torch.Tensor, phih: torch.Tensor) -> torch.Tensor:
        norm_loss = torch.mean((y - phih) ** 2)
        return self.alpha * norm_loss

    def preget(self, x: torch.Tensor) -> None:
        # x: [bs, input_sequence_length, num_features]
        x_t = x.permute(2, 0, 1)  # [num_features, bs, input_sequence_length]
        theta = torch.bmm(x_t, self.reduce_mlayer).permute(
            1, 2, 0
        )  # [bs, 2, num_features]
        theta = self.activation(theta)
        self.phil, self.phih = theta[:, :1, :], theta[:, 1:, :]
        self.xil = torch.sum(torch.pow(x - self.phil, 2), dim=1, keepdim=True) / (
            x.shape[1] - 1
        )
        self.xih = torch.sum(torch.pow(x - self.phih, 2), dim=1, keepdim=True) / (
            x.shape[1] - 1
        )

    def forward(
        self, x: torch.Tensor, mode: Literal["norm", "denorm"] = "norm"
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if mode == "norm":
            self.preget(x)
            x = (x - self.phil) / torch.sqrt(self.xil + 1e-8) * self.gamma + self.beta
            return x, self.phih
        elif mode == "denorm":
            x = ((x - self.beta) / self.gamma) * torch.sqrt(self.xih + 1e-8) + self.phih
        return x


def main() -> None:
    x = torch.randn((32, 336, 8), dtype=torch.float32)
    y = torch.randn((32, 192, 8), dtype=torch.float32)

    model = DishTS(336, 8, alpha=1.0)
    x_norm, phih = model.forward(x, mode="norm")
    print(x_norm.shape, phih.shape)

    y_denorm = model.forward(y, mode="denorm")
    print(y_denorm.shape)

    norm_loss = model.normalization_loss(y, phih)
    print(y, phih)
    print(norm_loss)


if __name__ == "__main__":
    main()
