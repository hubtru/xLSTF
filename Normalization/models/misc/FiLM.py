import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from Normalization.models import BaseModel
from Normalization.models.misc.FrequencyEnhancedLayer import FrequencyEnhancedLayer
from Normalization.models.normalization import LegendreProjectionUnit


class FiLM(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        modes1: int = 8,
        compression: int = 16,
        d_model: int = 16,
        ratio: float = 1.0,
        use_RevIN: bool = False,
        mode_type: int = 0,
        output_process_information: bool = False,
        **kwargs,
    ) -> None:
        super(FiLM, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )

        self.output_process_information = output_process_information
        self.modes1 = min(modes1, self.output_sequence_length // 2)

        self.enc_in = num_features
        self.dec_in = num_features

        self.projection_required = False
        self.mode_type = mode_type

        if num_features > 1000:
            self.projection_required = True
            self.in_projection = nn.Conv1d(self.enc_in, d_model, 1)
            self.out_projection = nn.Conv1d(d_model, self.dec_in, 1)
            self.d_model = d_model

        self.use_RevIN = use_RevIN
        if use_RevIN:
            if self.projection_required:
                self.affine_weight = nn.Parameter(
                    torch.ones((1, 1, d_model), dtype=torch.float32)
                )
                self.affine_bias = nn.Parameter(
                    torch.zeros((1, 1, d_model), dtype=torch.float32)
                )
            else:
                self.affine_weight = nn.Parameter(
                    torch.ones((1, 1, self.enc_in), dtype=torch.float32)
                )
                self.affine_bias = nn.Parameter(
                    torch.zeros((1, 1, self.enc_in), dtype=torch.float32)
                )
        else:
            pass

        self.multiscale = [1, 2, 4]
        self.window_size = [256]

        if (
            self.multiscale[-1] * self.output_sequence_length
        ) < self.input_sequence_length:
            logging.warning(
                f"Due to the multiscale approach of the model, some part of the lookback window will be ignored. Check that the lookback window is at most {self.multiscale[-1] * self.output_sequence_length} steps long."
            )

        self.legendre_projections = nn.ModuleList(
            [
                LegendreProjectionUnit(N=n, dt=1.0 / self.output_sequence_length / i)
                for n in self.window_size
                for i in self.multiscale
            ]
        )

        self.backbones = nn.ModuleList(
            [
                FrequencyEnhancedLayer(
                    in_channels=n,
                    out_channels=n,
                    sequence_length=min(
                        self.output_sequence_length, self.input_sequence_length
                    ),
                    modes1=modes1,
                    compression=compression,
                    ratio=ratio,
                    mode_type=self.mode_type,
                )
                for n in self.window_size
                for _ in range(len(self.multiscale))
            ]
        )
        self.mlp = nn.Linear(len(self.multiscale) * len(self.window_size), 1)

        self.ratio = ratio
        self.compression = compression

    def params(self) -> dict:
        return {
            "modes1": self.modes1,
            "compression": self.compression,
            "ratio": self.ratio,
            "useRevIN": self.use_RevIN,
            "modeType": self.mode_type,
        }

    def add_tensor_to_history(
        self, x: torch.Tensor, history: Optional[List[torch.Tensor]] = None
    ) -> None:
        if self.output_process_information:
            if history is None:
                history = []
            history.append(x)

    def forward(
        self,
        x_enc: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        history = [] if self.output_process_information else None
        self.add_tensor_to_history(x_enc, history)

        if self.projection_required:
            x_enc = self.in_projection(x_enc.transpose(1, 2))
            x_enc = x_enc.transpose(1, 2)

        if self.use_RevIN:
            mean = x_enc.mean(1, keepdim=True).detach()
            std = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x_enc = (x_enc - mean) / std
            x_enc = x_enc * self.affine_weight + self.affine_bias

        x_decs = []
        jump_dist = 0
        for i in range(0, len(self.multiscale) * len(self.window_size)):
            x_in_len = (
                self.multiscale[i % len(self.multiscale)] * self.output_sequence_length
            )
            x_in = x_enc[:, -x_in_len:]
            legendre_proj = self.legendre_projections[i]
            x_in_c = legendre_proj(x_in)[:, :, :, jump_dist:]
            out1 = self.backbones[i](x_in_c)
            out1 = out1.transpose(2, 3)
            if self.input_sequence_length >= self.output_sequence_length:
                x_dec_c = out1[:, :, self.output_sequence_length - 1 - jump_dist, :]
            else:
                x_dec_c = out1[:, :, -1, :]
            x_dec = (
                x_dec_c @ legendre_proj.eval_matrix[-self.output_sequence_length :, :].T
            )
            x_decs += [x_dec]

        self.add_tensor_to_history(x_in_c, history)
        self.add_tensor_to_history(out1, history)

        x_dec = self.mlp(torch.stack(x_decs, dim=-1)).squeeze(-1).permute(0, 2, 1)

        if self.use_RevIN:
            x_dec = (x_dec - self.affine_bias) / self.affine_weight
            x_dec = (x_dec * std) + mean

        if self.projection_required:
            x_dec = self.out_projection(x_dec.transpose(1, 2))
            x_dec = x_dec.transpose(1, 2)
        self.add_tensor_to_history(x_dec, history)
        return x_dec, history
