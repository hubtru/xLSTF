from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .utils import getitem


class ETTmxDataset(Dataset):
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
        if size is not None:
            self.sequence_length = size[0]
            self.label_length = size[1]
            self.prediction_length = size[2]
        else:
            self.sequence_length = 24 * 4 * 4
            self.label_length = 24 * 4
            self.prediction_length = 24 * 4

        assert flag in ["train", "val", "test"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.train_only = train_only

        self.root_dir = root_dir
        self.filename = filename
        self._load()

    def _load(self) -> None:
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.root_dir / self.filename)

        border1s = [
            0,
            12 * 30 * 24 * 4 - self.sequence_length,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.sequence_length,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
        border1 = border1s[self.type]
        border2 = border2s[self.type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            raise ValueError()

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]].values
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return getitem(self, idx)

    def __len__(self) -> int:
        return len(self.data_x) - self.sequence_length - self.prediction_length + 1

    def inverse_transform(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return self.scaler.inverse_transform(x)
