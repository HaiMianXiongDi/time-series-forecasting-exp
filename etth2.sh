#!/usr/bin/env bash
# run_etth2_multi_pred.sh
# 类似于你给的 ETTh1 风格脚本，但针对 ETTh2
# 遍历 pred_len in {336}，其余固定参数示例：
#   seq_len=96, batch_size=512, head_dropout=0.2, learning_rate=0.001, train_epochs=20
# 并命名 model_id 为 ETTh2_96_pl{pred_len}_bs512_d0.2_lr0.001_LPUNET

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# 你可以自行修改 pred_lens 数组，比如 (96 192 336 720)。这里演示只含 336
seq_len=96
pred_lens=(192)

bs=256
hd=0
lr=0.0005
epochs=20
model_name=LPUNET
data_name=ETTh2

for pl in "${pred_lens[@]}"
do
    model_id="ETTh2_${seq_len}_pl${pl}_bs${bs}_d${hd}_lr${lr}_LPUNET"

    echo ">>> Running ETTh2 with pred_len=$pl"
    echo "    model_id=$model_id"

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ETT \
      --data_path ETTh2.csv \
      --model_id "$model_id" \
      --model "$model_name" \
      --data "$data_name" \
      --features M \
      --seq_len "$seq_len" \
      --pred_len "$pl" \
      --enc_in 7 \
      --des 'Exp' \
      --head_dropout "$hd" \
      --train_epochs "$epochs" \
      --stage_num 4 \
      --itr 1 \
      --batch_size "$bs" \
      --learning_rate "$lr"

    echo ">>> Finished pred_len=$pl"
done

echo "=== All experiments done for ETTh2 multi pred-lens. ==="
