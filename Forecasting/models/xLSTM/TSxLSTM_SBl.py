from functools import partial

import torch
from torch import nn
from xlstm import (FeedForwardConfig, mLSTMBlockConfig, mLSTMLayerConfig,
                   sLSTMBlockConfig, sLSTMLayerConfig, xLSTMBlockStack,
                   xLSTMBlockStackConfig)

from ..BaseModel import BaseModel
from ..normalization import RevIN

SLSTM_BLOCK_CONFIG = sLSTMBlockConfig(
    slstm=sLSTMLayerConfig(
        backend="cuda" if torch.cuda.is_available() else "vanilla",
        conv1d_kernel_size=0, # 0 = no conv4 and swish
        num_heads=4,
        bias_init="powerlaw_blockdependent",
    ),
    feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
)

xlstm_conf_fn = partial(
    xLSTMBlockStackConfig,
    slstm_block=SLSTM_BLOCK_CONFIG,
    num_blocks=1,
    slstm_at=[0],
)

class TSxLSTM_SBl(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        use_RevIN: bool = True,
        dropout_prob: float = 0.0,
        **kwargs,
    ) -> None:
        super(TSxLSTM_SBl, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )

        self.dropout_prob = dropout_prob
        self.use_RevIN = use_RevIN
        self.normalize = False
        if self.use_RevIN:
            self.norm = RevIN(num_features=self.num_features)

        cfg = xlstm_conf_fn(
            context_length=num_features,
            embedding_dim=self.input_sequence_length,
            dropout=dropout_prob,
        )
        self.xlstm = xLSTMBlockStack(cfg)

        self.drop = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(self.input_sequence_length, self.output_sequence_length)

    def params(self) -> dict:
        return {
            "norm": {
                "type": str(self.norm.__class__.__name__),
                "params": self.norm.params(),
            }
            if self.use_RevIN
            else None,
            "backbone": {
                "type": "xLSTM",
                "params": {
                    "context_length": self.num_features,
                    "embedding_dim": self.input_sequence_length,
                    "dropout": self.dropout_prob,
                },
            },
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, input_length, num_features)
        if self.normalize:
            seq_last = x[:, -1:, :].detach()
            x = x - seq_last

        if self.use_RevIN:
            x = self.norm.forward(x, mode="norm")

        x = x.transpose(1, 2)  # (bs, num_features, input_length)
        xlstm_output = self.xlstm(x)  # (bs, num_features, input_length)
        xlstm_output = self.drop(xlstm_output)
        output = self.fc(xlstm_output)  # (bs, num_features, output_length)
        output = output.transpose(1, 2)  # (bs, output_length, num_features)

        if self.use_RevIN:
            output = self.norm(output, mode="denorm")

        if self.normalize:
            output = output + seq_last

        # output: (bs, output_length, num_features)
        return output