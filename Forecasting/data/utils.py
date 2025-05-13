from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .CustomDataset import CustomDataset
    from .ETThxDataset import ETThxDataset
    from .ETTmxDataset import ETTmxDataset
    from .ThreeColumnDataset import ThreeColumnDataset


def getitem(
    ds: Union["CustomDataset", "ETThxDataset", "ETTmxDataset", "ThreeColumnDataset"],
    idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    begin_input_sequence = idx
    end_input_sequence = begin_input_sequence + ds.sequence_length
    begin_gt_sequence = end_input_sequence - ds.label_length
    end_gt_sequence = begin_gt_sequence + ds.label_length + ds.prediction_length

    input_sequence = ds.data_x[begin_input_sequence:end_input_sequence]
    gt_sequence = ds.data_y[begin_gt_sequence:end_gt_sequence]

    return input_sequence, gt_sequence


def read_data(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Source: https://github.com/decisionintelligence/TFB/blob/master/ts_benchmark/data/utils.py"""
    data = pd.read_csv(path)
    label_exists = "label" in data["cols"].values

    all_points = data.shape[0]

    columns = data.columns

    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()

    is_univariate = n_points == all_points

    n_cols = all_points // n_points
    df = pd.DataFrame()

    cols_name = data["cols"].unique()

    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    else:
        df[cols_name[0]] = data.iloc[:, 0]

    if label_exists:
        # Get the column name of the last column
        last_col_name = df.columns[-1]
        # Renaming the last column as "label"
        df.rename(columns={last_col_name: "label"}, inplace=True)

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]

    return df
