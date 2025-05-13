#!/bin/bash
PYTHONPATH=$PYTHONPATH:/home/name.surname/xLSTF/Normalization

conda activate ml

dataset=$1
input_sequence_length=$2
output_sequence_length=$3
run_id=$4


echo "Run experiment ($dataset, Baseline) [$input_sequence_length, $output_sequence_length]"
python3.12 xLSTF/cli.py --dataset "$dataset" \
    --model "Baseline" \
    --lookback-window "$input_sequence_length" \
    --forecasting-horizon "$output_sequence_length" \
    --run-id "$run_id"

models=("Left" "Right" "Sandwich")
skip_connections=("skip" "no_skip")
hidden_factor=("1.0" "2.0")
activation_fn=("none" "gelu")

for model in "${models[@]}"; do
  for hidden in "${hidden_factor[@]}"; do
    for skip in "${skip_connections[@]}"; do
      for act_fn in "${activation_fn[@]}"; do
        echo "Run experiment ($dataset, $model) [$input_sequence_length, $output_sequence_length]"
        python3.12 xLSTF/cli.py --dataset "$dataset" \
            --model "$model" \
            --lookback-window "$input_sequence_length" \
            --forecasting-horizon "$output_sequence_length" \
            --hidden-factor "$hidden" \
            --activation "$act_fn" \
            --skip-connection "$skip" \
            --run-id "$run_id"
      done
    done
  done
done
