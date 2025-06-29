#!/usr/bin/env bash
# run_ettm2_4combos.sh
# 只跑以下4个 ETTm2 参数组合(参考你给的MSE/MAE结果):
# 1) ETTm2_sl96_pl96_bs128_hd0.5_lr0.0005_LPUNET
# 2) ETTm2_sl96_pl192_bs128_hd0.2_lr0.0001_LPUNET
# 3) ETTm2_sl96_pl336_bs128_hd0.3_lr0.0001_LPUNET
# 4) ETTm2_sl96_pl720_bs258_hd0.3_lr0.0001_LPUNET

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# 固定配置
model_name=LPUNET
dataset=ETTm2
root_path=./dataset/ETT
data_path=ETTm2.csv
seq_len=96
stage_num=4
train_epochs=20    # 若你只想跑20 epoch，可改此处
enc_in=7           # ETTm2 通常为7列
des=Exp
itr=1
features=M         # Multivariate

########################################
# 1) pl=96, bs=128, hd=0.5, lr=0.0005
########################################
pl=96
bs=128
hd=0.5
lr=0.0005
model_id="ETTm2_sl${seq_len}_pl${pl}_bs${bs}_hd${hd}_lr${lr}_LPUNET"
echo ">>> [1/4] $model_id"

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
  --learning_rate "$lr"

########################################
# 2) pl=192, bs=128, hd=0.2, lr=0.0001
########################################
pl=192
bs=128
hd=0.2
lr=0.0001
model_id="ETTm2_sl${seq_len}_pl${pl}_bs${bs}_hd${hd}_lr${lr}_LPUNET"
echo ">>> [2/4] $model_id"

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
  --learning_rate "$lr"

# ########################################
# # 3) pl=336, bs=128, hd=0.3, lr=0.0001
# ########################################
pl=336
bs=128
hd=0.3
lr=0.001
model_id="ETTm2_sl${seq_len}_pl${pl}_bs${bs}_hd${hd}_lr${lr}_LPUNET"
echo ">>> [3/4] $model_id"

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
  --learning_rate "$lr"

# ########################################
# # 4) pl=720, bs=258, hd=0.3, lr=0.0001
# ########################################
pl=720
bs=258
hd=0.3
lr=0.0001
model_id="ETTm2_sl${seq_len}_pl${pl}_bs${bs}_hd${hd}_lr${lr}_LPUNET"
echo ">>> [4/4] $model_id"

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
  --learning_rate "$lr"

echo "=== All 4 combos done for ETTm2 ==="
