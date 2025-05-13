import torch
from torch import nn

from ..BaseModel import BaseModel


class Linear(BaseModel):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        individual: bool = False,
        **kwargs,
    ) -> None:
        super(Linear, self).__init__(
            input_sequence_length,
            output_sequence_length,
            num_features,
        )
        self.individual = individual

        if self.individual:
            self.weights = nn.ModuleList()
            for i in range(self.num_features):
                self.weights.append(
                    nn.Linear(input_sequence_length, output_sequence_length)
                )
        else:
            self.weights = nn.Linear(input_sequence_length, output_sequence_length)

    def params(self) -> dict:
        return {"individual": self.individual}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.individual:
            output = torch.zeros(
                (x.shape[0], self.output_sequence_length, x.shape[2]),
                dtype=x.dtype,
                device=x.device,
            )
            for i in range(self.num_features):
                output[:, :, i] = self.weights[i](x[:, :, i])
            x = output
        else:
            x = self.weights(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x
