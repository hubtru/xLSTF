from typing import Callable, Optional, Tuple

import torch
from torch import nn


class FrequencyPredictionUnit(nn.Module):
    def __init__(
        self, lookback_window: int, forecasting_horizon: int, num_features: int
    ) -> None:
        super().__init__()
        self.lookback_window = lookback_window
        self.forecasting_horizon = forecasting_horizon
        self.num_features = num_features

        self.model_freq = nn.Sequential(nn.Linear(lookback_window, 64), nn.ReLU())
        self.model_all = nn.Sequential(
            nn.Linear(64 + lookback_window, 128),
            nn.ReLU(),
            nn.Linear(128, forecasting_horizon),
        )

    def forward(self, main_frequencies: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch_size, input_sequence_length, num_features = x.shape
        assert (
            x.shape
            == main_frequencies.shape
            == (batch_size, self.num_features, self.lookback_window)
        )
        input = torch.cat([self.model_freq(main_frequencies), x], dim=-1)
        assert input.shape == (batch_size, self.num_features, 64 + self.lookback_window)

        output = self.model_all(input)
        assert output.shape == (batch_size, self.num_features, self.forecasting_horizon)

        return output


def frequency_normalization(
    x: torch.Tensor, k: int, rfft: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, input_sequence_length, num_features = x.shape

    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)
    assert xf.shape == (batch_size, int(input_sequence_length // 2) + 1, num_features)

    k = min(k, xf.shape[1])
    k_values = torch.topk(torch.abs(xf), k=k, dim=1)
    indices = k_values.indices

    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask
    assert (
        xf_filtered.shape
        == xf.shape
        == (batch_size, int(input_sequence_length // 2) + 1, num_features)
    )

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()
    assert (
        x_filtered.shape == x.shape == (batch_size, input_sequence_length, num_features)
    )

    x_norm = x - x_filtered
    assert x_norm.shape == x.shape == (batch_size, input_sequence_length, num_features)

    return x_norm, x_filtered


class FAN(nn.Module):
    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        num_features: int,
        freq_topk: int = 20,
        rfft: bool = True,
    ) -> None:
        super().__init__()
        self.lookback_window = input_sequence_length
        self.forecasting_horizon = output_sequence_length
        self.num_features = num_features
        self.top_k_frequencies = freq_topk
        self.use_rfft = rfft
        self.epsilon: float = 1e-8

        self.model = FrequencyPredictionUnit(
            lookback_window=input_sequence_length,
            forecasting_horizon=output_sequence_length,
            num_features=num_features,
        )

    def params(self) -> dict:
        return {"use_rfft": self.use_rfft, "top_k_frequencies": self.top_k_frequencies}

    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, input_sequence_length, num_features = x.shape
        x_norm, x_filtered = frequency_normalization(
            x, k=self.top_k_frequencies, rfft=self.use_rfft
        )

        predicted_frequency_signal = self.model(
            x_filtered.transpose(1, 2), x.transpose(1, 2)
        ).transpose(1, 2)

        return (
            x_norm.reshape(batch_size, input_sequence_length, num_features),
            predicted_frequency_signal,
        )

    def denormalize(
        self, x: torch.Tensor, predicted_frequency_signal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, output_sequence_length, num_features = x.shape
        output = x + predicted_frequency_signal

        return output.reshape(batch_size, output_sequence_length, num_features), x

    def normalization_loss(
        self,
        y_true: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, output_sequence_length, num_features = y_true.shape
        residual, pred_main = frequency_normalization(
            y_true, k=self.top_k_frequencies, rfft=self.use_rfft
        )

        predicted_frequency_signal, predicted_residual = stats
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        return loss_fn(predicted_frequency_signal, pred_main) + loss_fn(
            predicted_residual, residual
        )
