from typing import Literal, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .utils import pre_train_stat_model

PATCH_SIZE_CONFIGS = {36: 6, 104: 4, 336: 12, 512: 16}


class SAN(nn.Module):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        patch_size: Optional[int] = None,
        feature_type: Literal["M", "S", "MS"] = "M",
    ) -> None:
        super(SAN, self).__init__()
        if patch_size is None:
            patch_size = PATCH_SIZE_CONFIGS[input_sequence_length]

        self.lookback_window = input_sequence_length
        self.forecasting_horizon = output_sequence_length
        self.num_features = num_features if feature_type == "M" else 1

        self.patch_length = patch_size
        self.num_patches_in = self.lookback_window // self.patch_length
        self.num_patches_out = self.forecasting_horizon // self.patch_length
        if self.num_patches_in * self.patch_length != self.lookback_window:
            raise ValueError("lookback_window must be divisible by patch_size")
        if self.num_patches_out * self.patch_length != self.forecasting_horizon:
            raise ValueError("forecasting_horizon must be divisible by patch_size")

        self.epsilon: float = 1e-5

        self.affine_weight = nn.Parameter(
            torch.ones((2, self.num_features), dtype=torch.float32), requires_grad=True
        )
        self.model = nn.ModuleDict(
            {
                "mean": StatisticsPredictionUnit(
                    num_patches_in=self.num_patches_in,
                    num_patches_out=self.num_patches_out,
                    patch_size=self.patch_length,
                    mode="mean",
                ),
                "std": StatisticsPredictionUnit(
                    num_patches_in=self.num_patches_in,
                    num_patches_out=self.num_patches_out,
                    patch_size=self.patch_length,
                    mode="std",
                ),
            }
        )

    def params(self) -> dict:
        return {
            "patch_size": self.patch_length,
            "num_patches_in": self.num_patches_in,
            "num_patches_out": self.num_patches_out,
        }

    def train_module(
        self,
        train_dl: DataLoader,
        features: Literal["M", "S", "MS"] = "M",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.train()
        opt = torch.optim.AdamW(self.parameters(), lr=0.0003)
        pre_train_stat_model(
            self,
            opt,
            train_dl,
            self.patch_length,
            self.forecasting_horizon,
            epochs=5,
            features=features,
            device=device,
        )
        for param in self.parameters():
            param.requires_grad = False

    def normalization_loss(
        self, y_true: torch.Tensor, stats: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        batch_size, output_sequence_length, num_features = y_true.shape

        y_true = y_true.reshape(
            batch_size, self.num_patches_out, self.patch_length, num_features
        )
        assert y_true.shape == (
            batch_size,
            self.num_patches_out,
            self.patch_length,
            num_features,
        )

        mean_true, std_true = (torch.mean(y_true, dim=2), torch.std(y_true, dim=2))
        assert mean_true.shape == (batch_size, self.num_patches_out, num_features)
        assert std_true.shape == (batch_size, self.num_patches_out, num_features)

        mean_pred, std_pred = stats
        assert mean_pred.shape == (batch_size, self.num_patches_out, num_features)
        assert std_pred.shape == (batch_size, self.num_patches_out, num_features)

        return F.mse_loss(mean_true, mean_pred) + F.mse_loss(std_true, std_pred)

    def forward(
        self,
        x: torch.Tensor,
        stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mode: Literal["norm", "denorm"] = "norm",
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]] | torch.Tensor:
        if mode == "norm":
            return self.normalize(x)
        elif mode == "denorm":
            assert stats is not None
            return self.denormalize(x, stats)
        else:
            raise NotImplementedError()

    def normalize(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, input_sequence_length, num_features = x.shape

        x = x.reshape(batch_size, self.num_patches_in, self.patch_length, num_features)
        assert x.shape == (
            batch_size,
            self.num_patches_in,
            self.patch_length,
            num_features,
        )

        mean = torch.mean(x, dim=-2, keepdim=True)
        std = torch.std(x, dim=-2, keepdim=True)
        assert mean.shape == (batch_size, self.num_patches_in, 1, num_features)
        assert std.shape == (batch_size, self.num_patches_in, 1, num_features)

        x_norm = (x - mean) / (std + self.epsilon)
        x_norm = x_norm.reshape(batch_size, input_sequence_length, num_features)
        assert x_norm.shape == (batch_size, input_sequence_length, num_features)

        x = x.reshape(batch_size, input_sequence_length, num_features)
        assert x.shape == (batch_size, input_sequence_length, num_features)

        mean_all = torch.mean(x, dim=1, keepdim=True)
        assert mean_all.shape == (batch_size, 1, num_features)

        outputs_mean = (
            self.model["mean"](mean.squeeze(2) - mean_all, x - mean_all)
            * self.affine_weight[0]
            + mean_all * self.affine_weight[1]
        )
        outputs_std = self.model["std"](std.squeeze(2), x)
        assert outputs_mean.shape == (batch_size, self.num_patches_out, num_features)
        assert outputs_std.shape == (batch_size, self.num_patches_out, num_features)

        return x_norm, (outputs_mean, outputs_std)

    def denormalize(
        self, x: torch.Tensor, stats: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        batch_size, output_sequence_length, num_features = x.shape
        outputs_mean, outputs_std = (stats[0].unsqueeze(-2), stats[1].unsqueeze(-2))
        assert outputs_mean.shape == (batch_size, self.num_patches_out, 1, num_features)
        assert outputs_std.shape == (batch_size, self.num_patches_out, 1, num_features)

        x = x.reshape(batch_size, self.num_patches_out, self.patch_length, num_features)
        assert x.shape == (
            batch_size,
            self.num_patches_out,
            self.patch_length,
            num_features,
        )

        x = x * (outputs_std + self.epsilon) + outputs_mean
        assert x.shape == (
            batch_size,
            self.num_patches_out,
            self.patch_length,
            num_features,
        )

        x = x.reshape(batch_size, output_sequence_length, num_features)
        assert x.shape == (batch_size, output_sequence_length, num_features)

        return x


class StatisticsPredictionUnit(nn.Module):
    def __init__(
        self,
        num_patches_in: int,
        num_patches_out: int,
        patch_size: int,
        mode: Literal["mean", "std"],
    ) -> None:
        super().__init__()
        self.num_patches_in = num_patches_in
        self.num_patches_out = num_patches_out
        self.patch_size = patch_size

        if mode == "std":
            self.final_activation = nn.ReLU()
        elif mode == "mean":
            self.final_activation = nn.Identity()
        else:
            raise NotImplementedError()

        self.input = nn.Linear(self.num_patches_in, 512)
        self.input_raw = nn.Linear(self.num_patches_in * self.patch_size, 512)
        self.intermediate_activation = nn.ReLU() if mode == "std" else nn.Tanh()
        self.output = nn.Linear(1024, self.num_patches_out)

    def forward(self, x: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        batch_size, input_sequence_length, num_features = x_raw.shape

        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        assert x.shape == (batch_size, num_features, self.num_patches_in)
        assert x_raw.shape == (batch_size, num_features, input_sequence_length)

        x, x_raw = self.input(x), self.input_raw(x_raw)
        assert x.shape == (batch_size, num_features, 512)
        assert x_raw.shape == (batch_size, num_features, 512)

        x = torch.cat([x, x_raw], dim=-1)
        assert x.shape == (batch_size, num_features, 1024)

        x = self.output(self.intermediate_activation(x))
        assert x.shape == (batch_size, num_features, self.num_patches_out)

        x = self.final_activation(x)
        assert x.shape == (batch_size, num_features, self.num_patches_out)

        x = x.permute(0, 2, 1)
        assert x.shape == (batch_size, self.num_patches_out, num_features)

        return x


def main() -> None:
    x = torch.randn((32, 96, 5), dtype=torch.float32)
    y_pred = torch.randn((32, 336, 5), dtype=torch.float32)

    model = SAN(
        input_sequence_length=96,
        output_sequence_length=336,
        num_features=5,
    )
    x_norm, stats = model.normalize(x)
    y_denorm = model.denormalize(y_pred, stats)
    print(x_norm.shape, y_denorm.shape)


if __name__ == "__main__":
    main()
