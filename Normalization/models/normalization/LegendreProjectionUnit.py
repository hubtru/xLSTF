from typing import Tuple

import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F


def get_projection_matrices(N: int) -> Tuple[np.ndarray, np.ndarray]:
    Q = np.arange(N, dtype=np.float64)
    R = (2 * Q + 1)[:, None]  # / theta
    j, i = np.meshgrid(Q, Q)
    A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
    B = (-1.0) ** Q[:, None] * R
    return A, B


class LegendreProjectionUnit(nn.Module):
    def __init__(self, N, dt: float = 1 / 512, discretization="bilinear"):
        super(LegendreProjectionUnit, self).__init__()
        self.N = N
        self.discretization = discretization

        A, B = get_projection_matrices(N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        # dt, discretization options
        A, B, _, _, _ = scipy.signal.cont2discrete(
            (A, B, C, D), dt=dt, method=discretization
        )

        B = B.squeeze(-1)

        self.register_buffer("A", torch.from_numpy(A).to(torch.float32))
        self.register_buffer("B", torch.from_numpy(B).to(torch.float32))
        vals = np.arange(0.0, 1.0, dt)
        self.register_buffer(
            "eval_matrix",
            torch.Tensor(
                scipy.special.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T
            ),
        )

    def params(self) -> dict:
        return {"num_polynomials": self.N, "discretization": self.discretization}

    def forward(
        self, inputs: torch.Tensor
    ) -> torch.Tensor:  # torch.Size([128, 1, 1]) -
        # inputs: [bs, seq_len, num_features]
        inputs = inputs.transpose(1, 2)  # [bs, num_features, seq_len]

        c = torch.zeros((inputs.shape[:-1] + tuple([self.N])), device=inputs.device)
        cs = []
        for f in inputs.permute([-1, 0, 1]):
            f = f.unsqueeze(-1)
            new = f @ self.B.unsqueeze(0)
            c = F.linear(c, self.A) + new
            cs.append(c)
        return torch.stack(cs, dim=0).permute(
            1, 2, 3, 0
        )  # [bs, num_features, N, seq_len]

    def reconstruct(self, c: torch.Tensor, seq_len: int, pred_len: int) -> torch.Tensor:
        assert self.eval_matrix.shape[0] == pred_len

        # c: [bs, num_features, N, seq_len]
        c = c.permute(0, 1, 3, 2)  # [bs, num_features, seq_len, N]

        if seq_len >= pred_len:
            c = c[:, :, pred_len - 1, :]  # [bs, num_features, N]
        else:
            c = c[:, :, -1, :]  # [bs, num_features, N]
        out = c @ self.eval_matrix[-pred_len:, :].T
        return out.transpose(-1, -2)


def main() -> None:
    x = torch.randn((32, 336, 7), dtype=torch.float32)
    model = LegendreProjectionUnit(N=256)
    out = model.forward(x)
    print(out.shape)
    rec = model.reconstruct(out, seq_len=336, pred_len=96)
    print(rec.shape)


if __name__ == "__main__":
    main()
