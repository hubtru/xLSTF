import logging
from typing import Literal, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from xLSTF.utils import flatten_dict


def stat_loss(
    y_gt: torch.Tensor,
    stat_predictions: Tuple[torch.Tensor, torch.Tensor],
    output_sequence_length: int,
    feature_dim: int,
    patch_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs, seq_len, num_features = y_gt.shape

    y_gt = y_gt[:, -output_sequence_length:, feature_dim:]
    y_gt = y_gt.reshape(bs, -1, patch_length, num_features)
    mean_gt, std_gt = torch.mean(y_gt, dim=2), torch.std(y_gt, dim=2)
    mean_pred, std_pred = stat_predictions

    mean_loss = F.mse_loss(mean_pred, mean_gt)
    std_loss = F.mse_loss(std_pred, std_gt)
    total_loss = mean_loss + std_loss

    return total_loss, mean_loss, std_loss


def pre_train_stat_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    patch_length: int,
    output_sequence_length: int,
    epochs: int = 100,
    features: Literal["MS", "S", "M"] = "MS",
    verbose: bool = False,
    device: torch.device = torch.device("cuda"),
) -> None:
    if verbose:
        logging.info("Start training of the statistics prediction model...")
    feature_dim = -1 if features == "MS" else 0

    for epoch in range(1, epochs + 1):
        losses = {"total": [], "mean": [], "std": []}
        for x, y in train_loader:
            x, y = x.float().to(device), y.float().to(device)
            optimizer.zero_grad()

            _, (mean_pred, std_pred) = model.normalize(x)
            total, mean, std = stat_loss(
                y,
                (mean_pred, std_pred),
                patch_length=patch_length,
                output_sequence_length=output_sequence_length,
                feature_dim=feature_dim,
            )
            total.backward()
            optimizer.step()

            losses["total"].append(total.item())
            losses["mean"].append(mean.item())
            losses["std"].append(std.item())

        if verbose:
            avg_dict = flatten_dict(losses, op="mean")
            logging.info(
                f"[{epoch:02d}|{epochs:02d}] - total loss: {avg_dict['total']:.7f} - mean component: {avg_dict['mean']:.7f} - std component: {avg_dict['std']:.7f}"
            )
