from typing import List, Literal, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


def pre_compute_covariance_matrices(
    train_dl: DataLoader,
    seq_len: int,
    pred_len: int,
    num_features: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], int]:
    cov_matrices = [
        torch.zeros((seq_len, pred_len), dtype=torch.float32, device=device)
        for _ in range(num_features)
    ]
    total_batches = 0

    for x, y in train_dl:
        x, y = (
            x.to(dtype=torch.float32, device=device),
            y.to(dtype=torch.float32, device=device),
        )
        total_batches = total_batches + 1

        for channel in range(num_features):
            x_c, y_c = x[..., channel], y[..., channel]
            cov_matrices[channel] = cov_matrices[channel] + torch.matmul(
                x_c.transpose(-1, -2), y_c[:, -pred_len:]
            )

    for channel in range(num_features):
        cov_matrices[channel] = cov_matrices[channel] / total_batches
    return cov_matrices, total_batches


def pre_compute_projection_matrices(
    cov_matrices: List[torch.Tensor], threshold: float = 0.05
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    u_list, v_list = [], []

    for channel in range(len(cov_matrices)):
        U, S, Vh = torch.linalg.svd(cov_matrices[channel])

        r = (S > threshold).sum().item()
        if r == 0:
            U_tilde = U[:, 0].unsqueeze(-1)
            V_tilde = Vh[0, :].unsqueeze(0).transpose(-1, -2)
        else:
            U_tilde = U[:, :r]
            V_tilde = Vh[:r, :].transpose(-1, -2)
        u_list.append(U_tilde)
        v_list.append(V_tilde)
    return u_list, v_list


class SIN(nn.Module):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        train_dl: DataLoader,
        threshold: float | None = None,
    ) -> None:
        super(SIN, self).__init__()
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.num_features = num_features

        self.threshold = (
            threshold
            if threshold
            else 0.05 * max(input_sequence_length, output_sequence_length)
        )
        self.scale = torch.sqrt(
            torch.tensor(
                (output_sequence_length / input_sequence_length), dtype=torch.float32
            )
        )
        cov_matrices, _ = pre_compute_covariance_matrices(
            train_dl,
            input_sequence_length,
            output_sequence_length,
            num_features,
            device=torch.device("cpu"),
        )
        u_list, v_list = pre_compute_projection_matrices(cov_matrices, self.threshold)

        for channel in range(self.num_features):
            self.register_buffer(
                f"u_{channel}", u_list[channel].to(dtype=torch.float32)
            )
            self.register_buffer(
                f"v_{channel}", v_list[channel].to(dtype=torch.float32)
            )
        self.phis = nn.ParameterList(
            [
                nn.Parameter(
                    torch.ones((u.shape[1],), dtype=torch.float32), requires_grad=True
                )
                for u in u_list
            ]
        )

    def params(self) -> dict:
        return {"threshold": self.threshold}

    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        _, _, C = x.shape

        theta_xs = []
        normalized_channels = []

        assert C == self.num_features
        for channel in range(self.num_features):
            x_c = x[..., channel]
            u_c = getattr(self, f"u_{channel}")

            theta_x_c = torch.matmul(x_c, u_c)
            theta_xs.append(theta_x_c)

            x_stat_c = torch.matmul(theta_x_c, u_c.transpose(-1, -2))
            x_norm_c = x_c - x_stat_c
            normalized_channels.append(x_norm_c)
        x_norm = torch.stack(normalized_channels, dim=-1)
        return x_norm, theta_xs

    def denormalize(
        self, x: torch.Tensor, theta_xs: List[torch.Tensor]
    ) -> torch.Tensor:
        restored_channels = []
        for channel in range(self.num_features):
            theta_x_c = theta_xs[channel]
            theta_y_c = theta_x_c.mul(self.phis[channel])

            v_c = getattr(self, f"v_{channel}")
            y_stat_c = torch.matmul(theta_y_c, v_c.transpose(-1, -2))

            y_hat_c = x[..., channel] + self.scale * y_stat_c
            restored_channels.append(y_hat_c)
        y_out = torch.stack(restored_channels, dim=-1)
        return y_out

    def forward(
        self,
        x: torch.Tensor,
        mode: Literal["norm", "denorm"],
        theta_x: torch.Tensor | None = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        match mode:
            case "norm":
                return self.normalize(x)
            case "denorm":
                if theta_x is None:
                    raise RuntimeError()
                return self.denormalize(x, theta_x)
            case _:
                raise RuntimeError()
