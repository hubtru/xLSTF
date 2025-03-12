from pathlib import Path
from typing import Literal, Optional, Tuple, Type, Union

import lightning.pytorch as L
from torch.utils.data import DataLoader

from Normalization.data import CustomDataset, ETThxDataset, ETTmxDataset
from Normalization.data.ThreeColumnDataset import ThreeColumnDataset


def get_dataset(
    dataset: str,
) -> Union[
    Type[ETTmxDataset],
    Type[ETThxDataset],
    Type[CustomDataset],
    Type[ThreeColumnDataset],
]:
    if dataset.lower() in ["etth1.csv", "etth2.csv"]:
        return ETThxDataset
    elif dataset.lower() in ["ettm1.csv", "ettm2.csv"]:
        return ETTmxDataset
    elif dataset.lower() in [
        "electricity.csv",
        "exchange_rate.csv",
        "national_illness.csv",
        "weather.csv",
        "traffic.csv",
        "reduced_traffic_12280.csv",
    ]:
        return CustomDataset
    elif dataset.lower() in [
        "aqshunyi.csv",
        "aqwan.csv",
        "covid-19.csv",
        "czelan.csv",
        "fred-md.csv",
        "metr-la.csv",
        "nasdaq.csv",
        "nn5.csv",
        "nyse.csv",
        "pems08.csv",
        "pems-bay.csv",
        "solar.csv",
        "wike2000.csv",
        "wind.csv",
        "zafnoo.csv",
    ]:
        return ThreeColumnDataset
    else:
        raise NotImplementedError()


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: Path,
        filename: str,
        batch_size: int,
        features: Literal["S", "M", "MS"] = "M",
        size: Optional[Tuple[int, int, int]] = None,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.filename = filename
        self.batch_size = batch_size
        self.size = size
        self.num_workers = num_workers

        ds_class = get_dataset(self.filename)
        self.train_ds = ds_class(self.root_dir, self.filename, self.size, flag="train")
        self.val_ds = ds_class(self.root_dir, self.filename, self.size, flag="val")
        self.test_ds = ds_class(self.root_dir, self.filename, self.size, flag="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )
