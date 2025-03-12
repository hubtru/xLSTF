from abc import abstractmethod
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from Normalization.models.utils import format_instance_str


class BaseModel(nn.Module):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        **kwargs,
    ) -> None:
        super(BaseModel, self).__init__()
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.num_features = num_features

    def get_instance_str(self) -> str:
        return format_instance_str(self, self.params())

    @abstractmethod
    def params(self) -> dict:
        pass

    def pretrain_model(
        self,
        train_dl: DataLoader,
        features: Literal["M", "S", "MS"] = "M",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
