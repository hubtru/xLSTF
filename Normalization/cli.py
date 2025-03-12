import argparse
import csv
import dataclasses
import time
from pathlib import Path
from typing import Optional, Tuple, Type

import lightning.pytorch as L
import torch
from lightning.pytorch.loggers import CSVLogger

from Normalization.callbacks import (LossCallback, ParameterCounterCallback)
from Normalization.data.DataModule import DataModule
from Normalization.model_wrapper import ModelWrapper
from Normalization.models import (BaseModel, FourierAnalysisNetwork, PreVsPostUp,
                            linear, misc, xLSTM)
from Normalization.utils import get_model_str

STANDARD_DATASETS = {
    "electricity.csv": 321,
    "ETTh1.csv": 7,
    "ETTh2.csv": 7,
    "ETTm1.csv": 7,
    "ETTm2.csv": 7,
    "exchange_rate.csv": 8,
    "national_illness.csv": 7,
    "reduced_traffic_12280.csv": 862,
    "Traffic.csv": 862,
    "weather.csv": 21,
}

ADDITIONAL_DATASETS = {
    "AQShunyi.csv": 11,
    "AQWan.csv": 11,
    "Covid-19.csv": 948,
    "CzeLan.csv": 11,
    "FRED-MD.csv": 107,
    "METR-LA.csv": 207,
    "NASDAQ.csv": 5,
    "NN5.csv": 111,
    "NYSE.csv": 5,
    "PEMS08.csv": 170,
    "PEMS-BAY.csv": 325,
    "Solar.csv": 137,
    "Wike2000.csv": 2000,
    "Wind.csv": 7,
    "ZafNoo.csv": 11,
}

FULL_DATASETS = STANDARD_DATASETS | ADDITIONAL_DATASETS


@dataclasses.dataclass
class ExperimentResult(object):
    name: str
    runtime: float
    data_loader: str
    des: str
    epochs: str
    features: str
    learning_rate: float
    model: str
    model_id: str
    pred_len: int
    seq_len: int
    normalization_method: str
    model_size: float
    num_parameters: int
    test_data_size: int
    test_loss: float
    training_data_size: int
    training_loss: float
    validation_data_size: int
    validation_loss: float
    ms_sample: str
    mae: float
    mse: float
    rmse: float


def load_model(name: str) -> Type[BaseModel]:
    model_class = (
        getattr(linear, name, None)
        or getattr(xLSTM, name, None)
        or getattr(misc, name, None)
        or getattr(PreVsPostUp, name, None)
        or getattr(FourierAnalysisNetwork, name, None)
    )
    if model_class is None:
        raise ValueError(f"Model {name} not found")
    return model_class


def setup_directories(base_dir: Optional[Path] = None) -> Tuple[Path, Path, Path, Path]:
    base_dir = base_dir if base_dir is not None else Path(__file__).parent
    logging_dir = base_dir / "Logs"
    raw_logging_dir = logging_dir / "raw"
    csv_dir = logging_dir / "csv_logs"
    data_dir = base_dir.parent / "Datasets"
    checkpoints_dir = base_dir / "Checkpoints"

    for path in [logging_dir, raw_logging_dir, csv_dir, checkpoints_dir]:
        if not path.exists():
            path.mkdir()
    return raw_logging_dir, csv_dir, data_dir, checkpoints_dir


def logs_experiment_results(
    result: ExperimentResult, concise_logging_file: Path
) -> None:
    if not concise_logging_file.exists():
        with open(concise_logging_file, "w") as f:
            writer = csv.writer(
                f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow([field.name for field in dataclasses.fields(result)])
    with open(concise_logging_file, "a") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(dataclasses.astuple(result))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=list(FULL_DATASETS.keys())
    )

    parser.add_argument("--hidden-factor", type=float, required=False, default=None)
    parser.add_argument(
        "--skip-connection",
        type=str,
        required=False,
        default=None,
        choices=["skip", "no_skip"],
    )
    parser.add_argument(
        "--activation",
        type=str,
        required=False,
        default="none",
        choices=["none", "gelu"],
    )
    parser.add_argument("--num-layers", type=int, required=False, default=None)

    parser.add_argument("--run-id", type=str, required=False, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--lookback-window", type=int, default=336)
    parser.add_argument("--label-length", type=int, default=None)
    parser.add_argument("--forecasting-horizon", type=int, default=192)
    parser.add_argument("--loss-fn", type=str, choices=["MAE", "MSE"], default="MAE")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda"
    )
    cfg = parser.parse_args()
    return cfg


def main() -> None:
    torch.set_float32_matmul_precision("high")

    (
        raw_logging_dir,
        csv_log_dir,
        data_dir,
        checkpoints_dir,
    ) = setup_directories()
    cfg = parse_arguments()

    model_params = {}
    if cfg.hidden_factor is not None:
        model_params["hidden_factor"] = int(cfg.hidden_factor)
    if cfg.skip_connection is not None:
        model_params["skip_connection"] = cfg.skip_connection
    if cfg.activation is not None:
        model_params["activation"] = cfg.activation
    if cfg.num_layers is not None:
        model_params["num_layers"] = cfg.num_layers

    if cfg.label_length is None:
        sizes = (cfg.lookback_window, cfg.lookback_window // 2, cfg.forecasting_horizon)
    else:
        sizes = (cfg.lookback_window, cfg.label_length, cfg.forecasting_horizon)

    data_module = DataModule(
        data_dir, cfg.dataset, size=sizes, batch_size=64, num_workers=4
    )

    model_params["train_dl"] = data_module.train_dataloader()

    model = load_model(cfg.model)(
        cfg.lookback_window,
        cfg.forecasting_horizon,
        FULL_DATASETS[cfg.dataset],
        **model_params,
    )

    wrapped_model = ModelWrapper(
        model,
        data_module.train_dataloader(),
        loss_fn="MAE",
        features="M",
    ).to(cfg.device)

    loss_cb = LossCallback()
    param_counter_cb = ParameterCounterCallback()
    early_stopping_cb = L.callbacks.EarlyStopping(
        monitor="val/MAE", patience=5, mode="min", min_delta=0.01
    )
    checkpointing_cb = L.callbacks.ModelCheckpoint(
        save_top_k=1, monitor="val/MAE", mode="min"
    )

    trainer = L.Trainer(
        max_epochs=100,
        num_sanity_val_steps=0,
        callbacks=[early_stopping_cb, checkpointing_cb, loss_cb, param_counter_cb],
        logger=[
            CSVLogger(
                raw_logging_dir,
                name=get_model_str(model.__class__.__name__, cfg.dataset, sizes),
            )
        ],
    )

    runtime_start = time.time()
    trainer.fit(wrapped_model, data_module)
    test_metrics = trainer.test(wrapped_model, data_module)
    overall_runtime = time.time() - runtime_start

    num_params, total_size = param_counter_cb.get_parameter_stats()

    results = ExperimentResult(
        name=get_model_str(cfg.model, cfg.dataset, sizes),
        runtime=round(overall_runtime, 2),
        data_loader=data_module.train_ds.__class__.__name__,
        des="",
        epochs=str(trainer.current_epoch),
        features="M",
        learning_rate=cfg.learning_rate,
        model=cfg.model,
        model_id=model.get_instance_str(),
        pred_len=cfg.forecasting_horizon,
        seq_len=cfg.lookback_window,
        normalization_method="",
        model_size=round(total_size, 2),
        num_parameters=num_params,
        test_data_size=len(data_module.test_ds),
        test_loss=round(test_metrics[0]["tst/MAE"], 3),
        training_data_size=len(data_module.train_ds),
        training_loss=round(loss_cb.get_loss("train"), 3),
        validation_data_size=len(data_module.val_ds),
        validation_loss=round(loss_cb.get_loss("val"), 3),
        ms_sample="",
        mae=round(test_metrics[0]["tst/MAE"], 3),
        mse=round(test_metrics[0]["tst/MSE"], 3),
        rmse=round(test_metrics[0]["tst/RMSE"], 3),
    )
    csv_log_file = (
        csv_log_dir / (cfg.run_id + ".csv")
        if cfg.run_id is not None
        else csv_log_dir / Path(get_model_str(cfg.model, cfg.dataset, sizes) + ".csv")
    )
    logs_experiment_results(results, csv_log_file)


if __name__ == "__main__":
    main()
