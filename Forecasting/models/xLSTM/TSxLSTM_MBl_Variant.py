from functools import partial
import numpy as np
import torch
from torch import nn

# NoConv = NoC import modified xlstm package, change to your own path location       
from ..xlstmMBlock.xlstm import (FeedForwardConfig, mLSTMBlockConfig, mLSTMLayerConfig,
                   sLSTMBlockConfig, sLSTMLayerConfig, xLSTMBlockStack,
                   xLSTMBlockStackConfig)

from ..BaseModel import BaseModel
from ..normalization import RevIN
from ..decomp import Linear_extractor

from ..linear import NLinear

MLSTM_BLOCK_CONFIG = mLSTMBlockConfig(
    mlstm=mLSTMLayerConfig(qkv_proj_blocksize=4, num_heads=4)
)

xlstm_conf_fn = partial(
    xLSTMBlockStackConfig,
    mlstm_block=MLSTM_BLOCK_CONFIG,
    num_blocks=1,
)

#DUET Linearpatternextractor
class LinearExtractorConfig:
    def __init__(self):
        self.seq_len = 104
        self.d_model = 104 
        self.moving_avg = 25
        self.enc_in = 1
        self.CI = 1

class TSxLSTM_MBl_Variant(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        use_RevIN: bool = True,
        dropout_prob: float = 0.0,
        **kwargs,
    ) -> None:
        super(TSxLSTM_MBl_Variant, self).__init__(
            input_sequence_length, output_sequence_length, num_features
        )

        self.dropout_prob = dropout_prob
        self.use_RevIN = use_RevIN
        self.normalize = False
        if self.use_RevIN:
            self.norm = RevIN(num_features=self.num_features)

        # Import Nlinear
        self.Linear1 = NLinear(
            input_sequence_length=self.input_sequence_length,
            output_sequence_length=self.input_sequence_length,
            num_features=self.num_features,
        )

        self.Linear2 = NLinear(
            input_sequence_length=self.input_sequence_length,
            output_sequence_length=self.input_sequence_length,
            num_features=self.num_features,
        )

        
        linear_extractor_config = LinearExtractorConfig()
        linear_extractor_config.seq_len = self.input_sequence_length
        linear_extractor_config.d_model = self.input_sequence_length
        self.linear_extractor = Linear_extractor(linear_extractor_config, individual=False)


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
            
        linear_output = self.Linear1(x)  # (bs, input_length, num_features)
        seasonal, trend = self.linear_extractor(linear_output)  # (bs, input_length, num_features)
        linear_output2 = self.Linear2(linear_output) # (bs, output_length, num_features)

        # Skip connection NLinear1
        skip_output = linear_output  # (bs, input_length, num_features)
        decomp = seasonal + trend
        # Add skip connection
        output = linear_output2 + skip_output # (bs,output_length, num_features)
        output = output.transpose(1, 2) + decomp
        # output = output.transpose(1, 2) 
        # Pass NLinear output to xLSTM
        xlstm_output = self.xlstm(output)  # (bs, num_features, output_length)
        xlstm_output = self.drop(xlstm_output)

        output = self.fc(xlstm_output)  # (bs, num_features, output_length)
        output = output.transpose(1, 2)  # (bs, output_length, num_features)

        if self.use_RevIN:
            output = self.norm(output, mode="denorm")

        if self.normalize:
            output = output + seq_last

        # output: (bs, output_length, num_features)
        return output