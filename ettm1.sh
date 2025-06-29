#!/usr/bin/env bash
# run_ettm1_multi_interp.sh
# 4 组固定超参 × interp_ratio ∈ {1.0 … 3.0}

model_name=LPUNET
dataset=ETTm1
root_path=./dataset
data_path=ETTm1.csv
seq_len=96
enc_in=7
stage_num=4
train_epochs=20
itr=1
des=Exp
features=M

# 需要遍历的插值倍率
interp_ratios=(1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0)

# 4 组固定组合 (pl  bs  dropout  lr)
declare -a combos=(
  "96  32 0 0.001"
  "192 32 0 0.001"
  "336 32 0 0.001"
  "720 32 0 0.001"
)

echo "=== ETTm1 4×interp_ratio experiments ==="
echo "interp_ratios={${interp_ratios[*]}}"

cid=1
for combo in "${combos[@]}"; do
  read -r pl bs hd lr <<< "${combo}"
  for interp_ratio in "${interp_ratios[@]}"; do
    model_id="ETTm1_sl${seq_len}_pl${pl}_bs${bs}_d${hd}_lr${lr}_interp${interp_ratio}_LPUNET"
    echo ">>> [$cid] pl=$pl bs=$bs d=$hd lr=$lr interp=$interp_ratio"

    python -u run_longExp.py \
      --is_training 1 \
      --root_path "$root_path" \
      --data_path "$data_path" \
      --model_id "$model_id" \
      --model "$model_name" \
      --data "$dataset" \
      --features "$features" \
      --seq_len "$seq_len" \
      --pred_len "$pl" \
      --enc_in "$enc_in" \
      --c_out "$enc_in" \
      --des "$des" \
      --stage_num "$stage_num" \
      --head_dropout "$hd" \
      --train_epochs "$train_epochs" \
      --itr "$itr" \
      --batch_size "$bs" \
      --learning_rate "$lr" \
      --interp_ratio "$interp_ratio"

    ((cid++))
  done
done

echo "=== All ETTm1 interp-ratio experiments finished ==="
