from typing import Literal

import torch
from torch import nn


class RevIN(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ) -> None:
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self._init_params()

    def params(self) -> dict:
        return {
            "eps": self.eps,
            "affine_weights": self.affine,
            "subtract_last": self.subtract_last,
        }

    def forward(self, x: torch.Tensor, mode: Literal["norm", "denorm"]) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self) -> None:
        self.affine_weight = nn.Parameter(
            torch.ones(self.num_features), requires_grad=True
        )
        self.affine_bias = nn.Parameter(
            torch.zeros(self.num_features), requires_grad=True
        )

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1:, :]
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.std = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.std
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.std
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


def main() -> None:
    x = torch.randn((48, 255, 10), dtype=torch.float32)
    model = RevIN(num_features=10, subtract_last=True)

    x_norm = model.forward(x, mode="norm")
    print(x_norm.shape)

    x_denorm = model.forward(x_norm, mode="denorm")
    print(x_denorm.shape)

    print(torch.isclose(x, x_denorm).all())


if __name__ == "__main__":
    main()
