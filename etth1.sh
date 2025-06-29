#!/usr/bin/env bash
# run_etth1_multi_interp.sh
# ETTh1 数据集：pred_len ∈ {96,192,336,720} × interp_ratio ∈ {1.0 … 3.0}

if [ ! -d "./logs" ]; then
  mkdir ./logs
fi
if [ ! -d "./logs/LongForecasting" ]; then
  mkdir ./logs/LongForecasting
fi

model_name=LPUNET
data_name=ETTh1
root_path=./dataset
data_path=ETTh1.csv

seq_len=336
enc_in=7
stage_num=4
itr=1
des=Exp

pred_lens=(720)
#interp_ratios=(1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0)
interp_ratios=(1.0)
batch_size=32
head_dropout=0
learning_rate=0.001
train_epochs=10

echo "=== ETTh1 experiments ==="
echo "pred_lens={${pred_lens[*]}}, interp_ratios={${interp_ratios[*]}}"

for pl in "${pred_lens[@]}"; do
  for interp_ratio in "${interp_ratios[@]}"; do
    model_id="ETTh1_${seq_len}_pl${pl}_bs${batch_size}_d${head_dropout}_lr${learning_rate}_interp${interp_ratio}_LPUNET"

    echo ">>> ETTh1: pl=$pl, interp=$interp_ratio"

    python -u run_longExp.py \
      --is_training 1 \
      --root_path "$root_path" \
      --data_path "$data_path" \
      --model_id "$model_id" \
      --model "$model_name" \
      --data "$data_name" \
      --features M \
      --seq_len "$seq_len" \
      --pred_len "$pl" \
      --enc_in "$enc_in" \
      --c_out "$enc_in" \
      --des "$des" \
      --stage_num "$stage_num" \
      --head_dropout "$head_dropout" \
      --train_epochs "$train_epochs" \
      --itr "$itr" \
      --batch_size "$batch_size" \
      --learning_rate "$learning_rate" \
      --interp_ratio "$interp_ratio"
  done
done

echo "=== ETTh1 experiments finished ==="
