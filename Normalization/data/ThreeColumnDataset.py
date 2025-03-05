from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset

from TSxLSTM.data.utils import getitem, read_data

DATASET_SPLITS = {
    "AQShunyi.csv": (0.6, 0.2, 0.2),
    "AQWan.csv": (0.6, 0.2, 0.2),
    "Covid-19.csv": (0.7, 0.1, 0.2),
    "CzeLan.csv": (0.7, 0.1, 0.2),
    "FRED-MD.csv": (0.7, 0.1, 0.2),
    "METR-LA.csv": (0.7, 0.1, 0.2),
    "NASDAQ.csv": (0.7, 0.1, 0.2),
    "NN5.csv": (0.7, 0.1, 0.2),
    "NYSE.csv": (0.7, 0.1, 0.2),
    "PEMS08.csv": (0.6, 0.2, 0.2),
    "PEMS-BAY.csv": (0.7, 0.1, 0.2),
    "Solar.csv": (0.6, 0.2, 0.2),
    "Traffic.csv": (0.7, 0.1, 0.2),
    "Wike2000.csv": (0.7, 0.1, 0.2),
    "Wind.csv": (0.7, 0.1, 0.2),
    "ZafNoo.csv": (0.7, 0.1, 0.2),
}


class ThreeColumnDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        filename: str,
        size: Optional[Tuple[int, int, int]] = None,
        flag: Literal["train", "val", "test"] = "train",
        scale: bool = True,
        splits: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        if size is None:
            self.sequence_length = 24 * 4 * 4
            self.label_length = 24 * 4
            self.prediction_length = 24 * 4
        else:
            self.sequence_length, self.label_length, self.prediction_length = size

        assert flag in ["train", "val", "test"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.type_ = type_map[flag]

        if splits is None:
            assert filename in DATASET_SPLITS.keys()
            splits = DATASET_SPLITS[filename]
        self.splits = splits
        self.scale = scale

        self.root_dir = root_dir
        self.filename = filename
        self._read_data()

    def _read_data(self) -> None:
        self.scaler = preprocessing.StandardScaler()
        file_path = self.root_dir / self.filename
        df = read_data(file_path)

        train_split, val_split, test_split = self.splits
        num_train = int(len(df) * train_split)
        num_test = int(len(df) * test_split)
        num_val = len(df) - num_train - num_test

        border1s = [
            0,
            num_train - self.sequence_length,
            len(df) - num_test - self.sequence_length,
        ]
        border2s = [num_train, num_train + num_val, len(df)]
        border1 = border1s[self.type_]
        border2 = border2s[self.type_]

        if self.scale:
            train_data = df.iloc[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df.values)
        else:
            data = df.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __len__(self) -> int:
        return len(self.data_x) - self.sequence_length - self.prediction_length + 1

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return getitem(self, idx)

    def inverse_transform(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
        return self.scaler.inverse_transform(x)
