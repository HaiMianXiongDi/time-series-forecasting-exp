#!/usr/bin/env bash
# run_traffic_multi_interp.sh
# Traffic 数据集：pred_len × interp_ratio(1.0–3.0)

model_name=LPUNET
data_name=custom
root_path=./dataset/traffic
data_path=traffic.csv

seq_len=96
train_epochs=20
stage_num=4
itr=1
des=Exp
enc_in=862        # Traffic 共 862 通道

# 固定超参
batch_size=32
learning_rate=0.0005
head_dropout=0.2

# 遍历列表
pred_lens=(96)    # 如需更多长度写成 (96 192 336 720)
interp_ratios=(1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0)

echo "=== Traffic experiments ==="
echo "batch=$batch_size  lr=$learning_rate  dropout=$head_dropout  epochs=$train_epochs"
echo "pred_lens={${pred_lens[*]}}, interp_ratios={${interp_ratios[*]}}"

for pl in "${pred_lens[@]}"; do
  for interp_ratio in "${interp_ratios[@]}"; do
    model_id="Traffic_sl${seq_len}_pl${pl}_bs${batch_size}_hd${head_dropout}_lr${learning_rate}_interp${interp_ratio}_LPUNET"

    echo ">>> Running Traffic: pl=$pl  interp=$interp_ratio"
    echo "    model_id=$model_id"

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

echo "=== Traffic interp-ratio experiments finished ==="
