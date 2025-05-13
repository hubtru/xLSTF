from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .utils import getitem


class CustomDataset(Dataset):
    """Source: https://github.com/cure-lab/LTSF-Linear/blob/main/data_provider/data_loader.py"""

    def __init__(
        self,
        root_dir: Path,
        filename: str,
        size: Tuple[int, int, int] | None = None,
        features: Literal["S", "M", "MS"] = "M",
        target: Literal["OT"] = "OT",
        flag: Literal["train", "val", "test"] = "train",
        scale: bool = True,
        train_only: bool = False,
    ) -> None:
        if size is None:
            self.sequence_length = 24 * 4 * 4
            self.label_length = 24 * 4
            self.prediction_length = 24 * 4
        else:
            self.sequence_length = size[0]
            self.label_length = size[1]
            self.prediction_length = size[2]

        assert flag in ["train", "val", "test"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.train_only = train_only

        self.root_dir = root_dir
        self.filename = filename
        self._load_data()

    def _load_data(self) -> None:
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.root_dir / self.filename)

        cols = list(df_raw.columns)
        if self.features == "S":
            cols.remove(self.target)
        cols.remove("date")

        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.2)
        num_validation = int(len(df_raw) - num_train - num_test)

        border1s = [
            0,
            num_train - self.sequence_length,
            len(df_raw) - num_test - self.sequence_length,
        ]
        border2s = [num_train, num_train + num_validation, len(df_raw)]
        border1 = border1s[self.type]
        border2 = border2s[self.type]

        if self.features == "M" or self.features == "MS":
            df_raw = df_raw[["date"] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_raw = df_raw[["date"] + cols + [self.target]]
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Features {self.features} not supported")

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return getitem(self, idx)

    def __len__(self) -> int:
        return len(self.data_x) - self.sequence_length - self.prediction_length + 1

    def inverse_transform(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return self.scaler.inverse_transform(x)
