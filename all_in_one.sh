#!/usr/bin/env bash
# run_elec_weather.sh
# 一次性跑 8 个实验：
# Part A: Electricity 数据 (pred_len in {96,192,336,720})
#          batch_size=32, head_dropout=0.2, lr=0.0005, 其余固定
# Part B: Weather    数据 (pred_len in {96,192,336,720})
#          batch_size=32, head_dropout=0.1, lr=0.001, 其余固定

#############################
# 公共超参 (若需改请自调)
#############################
seq_len=96
train_epochs=20
stage_num=4
itr=1
des=Exp

#############################
# Part A: Electricity
#############################
echo "======================"
echo "PHASE A: Electricity"
echo "======================"

# Electricity 通常 data=custom, root=./dataset/electricity, data_path=electricity.csv
# enc_in=321 (若你有别的列数,请自行修改)
E_root=./dataset/electricity
E_path=electricity.csv
E_data=custom
E_features=M
E_enc_in=321

# 该脚本参数
E_bs=32
E_hd=0.2
E_lr=0.0005
E_pred_lens=(96 192 336 720)

echo ">>> Electricity combos: batch_size=$E_bs, head_dropout=$E_hd, lr=$E_lr"
for pl in "${E_pred_lens[@]}"
do
  model_id="Electricity_${seq_len}_pl${pl}_bs${E_bs}_d${E_hd}_lr${E_lr}_LPUNET"

  echo "=== Electricity pred_len=$pl"
  echo "=== model_id=$model_id"

  python -u run_longExp.py \
    --is_training 1 \
    --root_path "$E_root" \
    --data_path "$E_path" \
    --model_id "$model_id" \
    --model LPUNET \
    --data "$E_data" \
    --features "$E_features" \
    --seq_len "$seq_len" \
    --pred_len "$pl" \
    --enc_in "$E_enc_in" \
    --c_out "$E_enc_in" \
    --des "$des" \
    --stage_num "$stage_num" \
    --head_dropout "$E_hd" \
    --train_epochs "$train_epochs" \
    --itr "$itr" \
    --batch_size "$E_bs" \
    --learning_rate "$E_lr"
done

echo ">>> Done all combos for Electricity."

#############################
# Part B: Weather
#############################
echo "======================"
echo "PHASE B: Weather"
echo "======================"

# Weather 数据： data=custom, root=./dataset, data_path=weather.csv
# enc_in=21 (若你有别的列数,请自行修改)
W_root=./dataset
W_path=weather.csv
W_data=custom
W_features=M
W_enc_in=21

# 该脚本参数
W_bs=32
W_hd=0.1
W_lr=0.001
W_pred_lens=(96 192 336 720)

echo ">>> Weather combos: batch_size=$W_bs, head_dropout=$W_hd, lr=$W_lr"
for pl in "${W_pred_lens[@]}"
do
  model_id="LPUNET_${seq_len}_pl${pl}_bs${W_bs}_d${W_hd}_lr${W_lr}_LPUNET"

  echo "=== Weather pred_len=$pl"
  echo "=== model_id=$model_id"

  python -u run_longExp.py \
    --is_training 1 \
    --root_path "$W_root" \
    --data_path "$W_path" \
    --model_id "$model_id" \
    --model LPUNET \
    --data "$W_data" \
    --features "$W_features" \
    --seq_len "$seq_len" \
    --pred_len "$pl" \
    --enc_in "$W_enc_in" \
    --c_out "$W_enc_in" \
    --des "$des" \
    --stage_num "$stage_num" \
    --head_dropout "$W_hd" \
    --train_epochs "$train_epochs" \
    --itr "$itr" \
    --batch_size "$W_bs" \
    --learning_rate "$W_lr"
done

echo ">>> Done all combos for Weather."
echo "=== All experiments (Electricity & Weather) finished! ==="
