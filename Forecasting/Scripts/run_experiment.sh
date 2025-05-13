#!/bin/bash
PYTHONPATH=$PYTHONPATH:/home/name.surname/xLSTF/Normalization
export PYTHONPATH

conda activate ml

models=("DLinear" "NLinear" "RLinear" "SANLinear" "SINLinear" "FANLinear" "DishLinear" "DFAN" "NFAN" "RFAN" "SANFAN" "SINFAN" "FANFAN" "DishFAN")

run_id=$1
dataset=$2
input_sequence_length=$3
output_input_sequence_length=$4

for model in "${models[@]}"; do
    echo "Run experiment ($dataset, $model) [$input_sequence_length, $output_input_sequence_length]"
    python3.12 xLSTF/cli.py --dataset "$dataset" --model "$model" --lookback-window "$input_sequence_length" --forecasting-horizon "$output_input_sequence_length" --run-id "$run_id"
done
