import numpy as np
import torch
from torch import nn


class FrequencyEnhancedLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sequence_length: int,
        modes1: int,
        compression: int = 0,
        ratio: float = 0.5,
        mode_type: int = 0,
    ) -> None:
        super(FrequencyEnhancedLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.compression = compression
        self.ratio = ratio
        self.mode_type = mode_type
        if self.mode_type == 1:
            # modes2=modes1-10000
            modes2 = modes1
            self.modes2 = min(modes2, sequence_length // 2)
            self.index0 = list(range(0, int(ratio * min(sequence_length // 2, modes2))))
            self.index1 = list(range(len(self.index0), self.modes2))
            np.random.shuffle(self.index1)
            self.index1 = self.index1[
                : min(sequence_length // 2, self.modes2)
                - int(ratio * min(sequence_length // 2, modes2))
            ]
            self.index = self.index0 + self.index1
            self.index.sort()
        elif self.mode_type > 1:
            # modes2=modes1-1000
            modes2 = modes1
            self.modes2 = min(modes2, sequence_length // 2)
            self.index = list(range(0, sequence_length // 2))
            np.random.shuffle(self.index)
            self.index = self.index[: self.modes2]
        else:
            self.modes2 = min(modes1, sequence_length // 2)
            self.index = list(range(0, self.modes2))

        self.scale = 1 / (in_channels * out_channels)
        if self.compression > 0:
            self.weights0 = nn.Parameter(
                self.scale
                * torch.rand(in_channels, self.compression, dtype=torch.cfloat)
            )
            self.weights1 = nn.Parameter(
                self.scale
                * torch.rand(
                    self.compression,
                    self.compression,
                    len(self.index),
                    dtype=torch.cfloat,
                )
            )
            self.weights2 = nn.Parameter(
                self.scale
                * torch.rand(self.compression, out_channels, dtype=torch.cfloat)
            )
        else:
            self.weights = nn.Parameter(
                self.scale
                * torch.randn(
                    (in_channels, out_channels, len(self.index)), dtype=torch.cfloat
                )
            )

    def forward(self, x):
        B, H, E, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            B,
            H,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        if self.compression == 0:
            if self.modes1 > 1000:
                for wi, i in enumerate(self.index):
                    out_ft[:, :, :, i] = torch.einsum(
                        "bji,io->bjo", (x_ft[:, :, :, i], self.weights1[:, :, wi])
                    )
            else:
                a = x_ft[:, :, :, : self.modes2]
                out_ft[:, :, :, : self.modes2] = torch.einsum(
                    "bjix,iox->bjox", a, self.weights
                )
        elif self.compression > 0:
            a = x_ft[:, :, :, : self.modes2]
            a = torch.einsum("bjix,ih->bjhx", a, self.weights0)
            a = torch.einsum("bjhx,hkx->bjkx", a, self.weights1)
            out_ft[:, :, :, : self.modes2] = torch.einsum(
                "bjkx,ko->bjox", a, self.weights2
            )
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
