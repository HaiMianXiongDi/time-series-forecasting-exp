#!/usr/bin/env bash
# run_exchange_pl192_gridsearch.sh
# Exchange 数据集预测长度固定为192，遍历以下超参数：
# bs={128,32,256,512}, hd={0.2,0.3,0.5}, lr={0.001,0.0005,0.0001}

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=LPUNET
data_name=custom
root_path=./dataset
data_path=exchange_rate.csv
features=M
enc_in=8        # Exchange 数据集通常为8维，如不同请自改
seq_len=96
pred_len=192
train_epochs=20
stage_num=4
itr=1
des=Exp

# 遍历的参数组合
batches=(128 32 256 512)
dropouts=(0.2 0.3 0.5)
lrs=(0.001 0.0005 0.0001)

echo "=== Exchange Dataset Grid Search ==="
echo "seq_len=$seq_len, pred_len=$pred_len, epochs=$train_epochs"

for bs in "${batches[@]}"
do
  for hd in "${dropouts[@]}"
  do
    for lr in "${lrs[@]}"
    do
      model_id="Exchange_sl${seq_len}_pl${pred_len}_bs${bs}_hd${hd}_lr${lr}_LPUNET"
      echo "Running: $model_id"

      python -u run_longExp.py \
        --is_training 1 \
        --root_path "$root_path" \
        --data_path "$data_path" \
        --model_id "$model_id" \
        --model "$model_name" \
        --data "$data_name" \
        --features "$features" \
        --seq_len "$seq_len" \
        --pred_len "$pred_len" \
        --enc_in "$enc_in" \
        --c_out "$enc_in" \
        --des "$des" \
        --stage_num "$stage_num" \
        --head_dropout "$hd" \
        --train_epochs "$train_epochs" \
        --itr "$itr" \
        --batch_size "$bs" \
        --learning_rate "$lr"

      echo "Finished: $model_id"
    done
  done
done

echo "=== All Exchange experiments completed! ==="
