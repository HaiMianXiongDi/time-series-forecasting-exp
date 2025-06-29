#!/usr/bin/env bash
# run_electricity_multi_interp.sh
# Electricity 数据集：pred_len ∈ {96,192,336,720} × interp_ratio ∈ {1.0 … 3.0}

model_name=LPUNET
root_path=./dataset
data_path=electricity.csv
data_name=custom
features=M
seq_len=96
enc_in=321      # 电力数据列
stage_num=4
train_epochs=10
itr=1
des=Exp

# 遍历参数
pred_lens=(96)
interp_ratios=(1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0)
head_dropout=0
batch_size=128
learning_rate=0.001

echo "=== Electricity experiments ==="
echo "pred_lens={${pred_lens[*]}}, interp_ratios={${interp_ratios[*]}}"

for pl in "${pred_lens[@]}"; do
  for interp_ratio in "${interp_ratios[@]}"; do
    model_id="Elec_sl${seq_len}_pl${pl}_bs${batch_size}_d${head_dropout}_lr${learning_rate}_interp${interp_ratio}_LPUNET"

    echo ">>> Running Electricity: pl=$pl, interp=$interp_ratio"

    python -u run_longExp.py \
      --is_training 1 \
      --root_path "$root_path" \
      --data_path "$data_path" \
      --model_id "$model_id" \
      --model "$model_name" \
      --data "$data_name" \
      --features "$features" \
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

echo "=== Electricity experiments finished ==="
