# xLSTF Normalization

## Description of the Project Structure
The `models` subdirectory contains four python module with model definitions. Starting with the `normalization` module,
this defines all normalization the definition of all normalizations evaluated and that can be used in models.
Meanwhile, the `linear`, `fan` and `xlstm` modules host the definitions from the actual forecasting models (those use the normalization methods define in the `normalization` module).

All models that are not easily classifiable into one of the model categories (`linear`, `fan` and `xlstm`) are stored in
the `misc` module.

The models used in the Pre- Vs. Post-Up experiment are stored in their own sub-module.


## Setup

### 1. Get the Datasets
All csv files must be placed directly into the `./Datasets` directory.
There are two different sources, one for the standard datasets (e.g. ETT benchmark, etc.) and the other one for the remaining datasets (like PEMS-BAY, METR-LA, etc.):
1. The [LSTF-Linear](https://github.com/cure-lab/LTSF-Linear) repository (see Getting Started -> Data Preparation)
2. The [TFB](https://github.com/decisionintelligence/TFB) repository (see Quickstart -> Data preparation)<br/>

### 2. Setup the Environment
The project’s dependencies are listed in the `pyproject.toml` file and can be easily installed using a Python project management tool. For example, by running the `uv sync` command with `uv`.

### 3. Add the project to the `PYTHONPATH` environmental variable
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/xLSTF/Normalization
```
To ensure that the Python interpreter can locate the source code of the project, add the absolute path to the `Normalization` directory to the `PYTHONPATH` environment variable.

## Usage
The project offers two recommended usage patterns. The first method trains a specified model on a single dataset using a defined configuration, while the second method employs shell scripts to evaluate the specified model across all available datasets and configurations.

### 1. Run the training script
```bash
python3 Normalization/cli.py --dataset <dataset-name:str> --model <model-name:str> --lookback-window <seq_len:int> --forecasting-horizon <pred_len:int>
```

### 2. Evaluate multiple models using the evaluation scripts
Sequentially evaluating multiple models can be done using the scripts provided in the `Scripts` directory.
The `run_LocalExperiment.sh` evaluates the provided model on all standard datasets with lookback-window of 336/104 and their corresponding forecasting horizons
```bash
./Normalization/Scripts/run_LocalExperiment.sh <model-name:str> <dataset-selector:str>
```
The `dataset-selector` has only three valid value `all`, `standard`, and `extended`. With `all` meaning that all datasets of the TFB benchmark will be used for evaluation, `standard` meaning that only the standard datasets are used for evaluation (e.g. ETT, Traffic, etc.), and `extended` meaning that only the remaining datasets (e.g. PEMS-BAY, METR-LA, etc.).<br/>

After the scripts has finished, the results of the run can be found in the `xLSTF/Logs/csv_logs/<dataset>.csv` file.
If all models should be evaluated (or only the models from certain experiments), there are special tags to pass to the `run_LocalExperiment.sh` scripts
```bash
./Normalization/Scripts/run_LocalExperiemnt.sh all <dataset-selector:str> # this evaluates all models
./Normalization/Scripts/run_LocalExperiemnt.sh linear <dataset-selector:str> # this evaluates only the linear models
./Normalization/Scripts/run_LocalExperiemnt.sh fan <dataset-selector:str> # this evaluates only the fourier analysis models
./Normalization/Scripts/run_LocalExperiemnt.sh xlstm <dataset-selector:str> # this evaluates all xlstm-based models

```
