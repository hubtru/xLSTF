#!/bin/bash
conda activate ml

model_selector=$1
dataset_selector=$2

if [ "$model_selector" = "all" ]
then
  models_to_evaluate=("DLinear" "NLinear" "RLinear" "FANLinear" "SANLinear" "SINLinear" "DishLinear" "LegendreLinear" "FiLM" "DFAN" "NFAN" "RFAN" "SANFAN" "FANFAN" "SINFAN" "DishFAN" "xLSTF" "SAN_xLSTF" "SIN_xLSTF" "FAN_xLSTF" "xLSTMMixer")
else
  if [ "$model_selector" = "linear" ]
  then
    models_to_evaluate=("DLinear" "NLinear" "RLinear" "FANLinear" "SANLinear" "SINLinear" "DishLinear" "LegendreLinear" "FiLM")
  else
    if [ "$model_selector" = "fan" ]
    then
      models_to_evaluate=("DFAN" "NFAN" "RFAN" "SANFAN" "FANFAN" "SINFAN" "DishFAN")
    else
      if [ "$model_selector" = "xlstm" ]
      then
        models_to_evaluate=("xLSTF" "TSxLSTM_MBl" "TSxLSTM_MBl_Variant" "TSxLSTM_SBl" "TSxLSTM_SBl_Variant")
      else
        models_to_evaluate=("$1")
      fi
    fi
  fi
fi

if [ "$dataset_selector" = "all" ]
then
  long_datasets=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather" "electricity" "Traffic" "Solar" "PEMS-BAY" "METR-LA" "PEMS08" "AQShunyi" "AQWan" "Wind" "ZafNoo" "CzeLan")
  short_datasets=("national_illness" "Covid-19" "NASDAQ" "NYSE" "FRED-MD" "NN5" "Wike2000")
else
  if [ "$dataset_selector" = "standard" ]
  then
    long_datasets=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather" "electricity" "Traffic")
    short_datasets=("national_illness")
  else
    if [ "$dataset_selector" = "extended" ]
    then
      long_datasets=("Traffic" "Solar" "PEMS-BAY" "METR-LA" "PEMS08" "AQShunyi" "AQWan" "Wind" "ZafNoo" "CzeLan")
      short_datasets=("Covid-19" "NASDAQ" "NYSE" "FRED-MD" "NN5" "Wike2000")
    fi
  fi
fi


long_pred_lens=(96 192 336 720)
for dataset in "${long_datasets[@]}"; do
  for model in "${models_to_evaluate[@]}"; do
    for pred_len in "${long_pred_lens[@]}"; do
      echo "Run experiment ($dataset, $model) [336, $pred_len]"
      python3.12 xLSTF/cli.py --dataset "$dataset.csv" --model "$model" --lookback-window "336" --forecasting-horizon "$pred_len" --run-id "${dataset}_EXP"
    done
  done
done

short_pred_lens=(24 36 48 60)
for dataset in "${short_datasets[@]}"; do
  for model in "${models_to_evaluate[@]}"; do
    for pred_len in "${short_pred_lens[@]}"; do
      echo "Run experiment ($dataset, $model) [104, $pred_len]"
      python3.12 xLSTF/cli.py --dataset "$dataset.csv" --model "$model" --lookback-window "104" --forecasting-horizon "$pred_len" --run-id "${dataset}_EXP.csv"
    done
  done
done
