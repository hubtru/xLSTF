from typing import Literal

import torch
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from xlstm import (mLSTMBlockConfig, sLSTMBlockConfig, sLSTMLayerConfig,
                   xLSTMBlockStack, xLSTMBlockStackConfig)

from xLSTF.models import BaseModel
from xLSTF.models.linear import DLinear, NLinear
from xLSTF.models.normalization import RevIN
from xLSTF.models.utils import SeriesDecomposition


class xLSTMMixer(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        xlstm_embedding_dim: int = 256,
        num_mem_tokens: int = 0,
        num_tokens_per_variate: int = 1,
        xlstm_dropout: float = 0.1,
        xlstm_conv1d_kernel_size: int = 0,
        xlstm_num_heads: int = 8,
        xlstm_num_blocks: int = 1,
        backcast: bool = True,
        packing: int = 1,
        backbone: Literal["NLinear", "DLinear"] = "NLinear",
        **kwargs,
    ) -> None:
        super().__init__(
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            num_features=num_features,
        )

        self.xlstm_embedding_dim = xlstm_embedding_dim
        self.mem_tokens = (
            nn.Parameter(torch.randn(num_mem_tokens, self.xlstm_embedding_dim) * 0.01)
            if num_mem_tokens > 0
            else None
        )
        slstm_config = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                num_heads=xlstm_num_heads, conv1d_kernel_size=xlstm_conv1d_kernel_size
            )
        )

        self.backbone = backbone
        self.backcast = backcast
        self.packing = packing

        if self.backbone == "NLinear":
            self.time_mixer = NLinear(
                self.input_sequence_length,
                self.output_sequence_length,
                self.num_features,
                individual=False,
            )
        elif self.backbone == "DLinear":
            self.time_mixer = DLinear(
                self.input_sequence_length,
                self.output_sequence_length,
                self.num_features,
                individual=False,
            )

        self.pre_encoding = nn.Linear(
            self.output_sequence_length, self.xlstm_embedding_dim
        )

        self.xlstm = xLSTMBlockStack(
            xLSTMBlockStackConfig(
                mlstm_block=(mLSTMBlockConfig()),
                slstm_block=slstm_config,
                num_blocks=xlstm_num_blocks,
                embedding_dim=self.xlstm_embedding_dim * self.packing,
                add_post_blocks_norm=True,
                dropout=xlstm_dropout,
                bias=True,
                slstm_at="all",
                context_length=self.num_features * num_tokens_per_variate
                + num_mem_tokens,
            )
        )

        if self.backcast:
            self.fc = nn.Linear(
                self.xlstm_embedding_dim * 2, self.output_sequence_length
            )
        else:
            self.fc = nn.Linear(self.xlstm_embedding_dim, self.output_sequence_length)

        self.norm = RevIN(num_features, affine=False)

        self.decomposition = SeriesDecomposition(25)
        self.seq_var_2_var_seq = Rearrange("batch seq var -> batch var seq")
        self.var_seq_2_seq_var = Rearrange("batch var seq -> batch seq var")

        (
            self.xlstm_dropout,
            self.xlstm_num_heads,
            self.xlstm_num_blocks,
            self.num_tokens_per_variate,
        ) = (xlstm_dropout, xlstm_num_heads, xlstm_num_blocks, num_tokens_per_variate)

    def params(self) -> dict:
        return {
            "norm": {
                "type": str(self.norm.__class__.__name__),
                "params": self.norm.params(),
            },
            "time_mixer": {
                "type": str(self.time_mixer.__class__.__name__),
                "params": self.time_mixer.params(),
            },
            "joint_mixer": {
                "type": "xLSTM",
                "params": {
                    "xlstm_embedding_dim": self.xlstm_embedding_dim,
                    "xlstm_dropout": self.xlstm_dropout,
                    "xlstm_num_heads": self.xlstm_num_heads,
                    "xlstm_num_blocks": self.xlstm_num_blocks,
                    "num_tokens_per_variate": self.num_tokens_per_variate,
                    "backcast": self.backcast,
                    "packing": self.packing,
                },
            },
        }

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        # norm needs b seq var
        x_enc = self.norm(x_enc, "norm")

        x_enc = self.time_mixer(x_enc)
        x_pre_forecast = self.seq_var_2_var_seq(x_enc)

        x = self.pre_encoding(x_pre_forecast)

        if self.packing > 1:
            var = x.shape[1]
            assert (
                var % self.packing == 0
            ), "The number of variables must be divisible by n"

            # Pack variables into sequence
            x = rearrange(x, "b (n var) seq -> b var (seq n)", n=self.packing)

        if self.mem_tokens is not None:
            m: Tensor = repeat(self.mem_tokens, "m d -> b m d", b=x.shape[0])
            x, mem_ps = pack([m, x], "b * d")

        dim = -1
        if self.backcast:
            x_reversed = torch.flip(x, [dim])
            x_bwd = self.xlstm(x_reversed)

        x_ = self.xlstm(x)
        x = x_

        if self.backcast:
            x = torch.cat((x, x_bwd), dim=dim)

        if self.mem_tokens is not None:
            m, x = unpack(x, mem_ps, "b * d")

        if self.packing > 1:
            x = rearrange(x, "b var (seq n) -> b (var n) seq", n=self.packing)

        x = self.fc(x)

        x = rearrange(x, "b v seq -> b seq v")
        # norm needs b seq var
        x = self.norm(x, "denorm")

        return x
