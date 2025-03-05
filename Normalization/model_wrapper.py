from typing import Any, Literal, Tuple

import lightning.pytorch as L
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from xLSTF.metrics import AVAILABLE_METRICS
from xLSTF.models import BaseModel
from xLSTF.models.normalization import FAN, DishTS

LOSS_FN = {"MSE": F.mse_loss, "MAE": F.l1_loss}


class ModelWrapper(L.LightningModule):
    def __init__(
        self,
        model: BaseModel,
        train_dl: DataLoader,
        learning_rate: float = 0.0003,
        loss_fn: Literal["MSE", "MAE"] = "MAE",
        features: Literal["M", "S", "MS"] = "M",
    ) -> None:
        super().__init__()
        self.model = model
        self.train_dl = train_dl
        self.learning_rate = learning_rate
        self.loss_fn = LOSS_FN.get(loss_fn)

        self.features = features
        self.feature_dim = -1 if features == "MS" else 0

    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def on_train_start(self) -> None:
        self.model.pretrain_model(self.train_dl, self.features, self.device)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch[0].float(), batch[1].float()

        out = self.model.forward(x)
        if isinstance(out, tuple):
            out, norm_vals = out

        y = y[:, -self.model.output_sequence_length :, self.feature_dim :]
        out = out[:, -self.model.output_sequence_length :, self.feature_dim :]
        if hasattr(self.model, "norm") and isinstance(self.model.norm, FAN):
            loss = self.model.norm.normalization_loss(
                y, norm_vals, self.loss_fn
            ) + self.loss_fn(out, y)
        elif hasattr(self.model, "norm") and isinstance(self.model.norm, DishTS):
            loss = self.model.norm.normalization_loss(y, norm_vals) + self.loss_fn(
                out, y
            )
        else:
            loss = self.loss_fn(out, y)

        self.log("train/loss", loss.item())
        return loss

    def _validation(
        self, batch: Tuple[torch.Tensor, torch.Tensor], prefix: Literal["val", "tst"]
    ) -> None:
        x, y = batch[0].float(), batch[1].float()

        out = self.model.forward(x)
        if isinstance(out, tuple):
            out, _ = out

        y = y[:, -self.model.output_sequence_length :, self.feature_dim :]
        out = out[:, -self.model.output_sequence_length :, self.feature_dim :]

        for name, metric in AVAILABLE_METRICS.items():
            self.log(f"{prefix}/{name}", metric(y, out))

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        self._validation(batch, "val")

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        self._validation(batch, "tst")
