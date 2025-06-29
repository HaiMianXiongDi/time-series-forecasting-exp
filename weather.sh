#!/usr/bin/env bash
# run_weather_multi_pred.sh
# 针对 Weather 数据集，遍历 pred_len in {96,192,336,720} 和 interp_ratio ∈ {1.0,1.5,2.0,2.5,3.0}

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# 固定配置
model_name=LPUNET
data_name=custom
root_path=./dataset
data_path=weather.csv
features=M
enc_in=21
seq_len=96
train_epochs=20
stage_num=4
itr=1
des=Exp

# 遍历的 pred_len 和 interp_ratio
pred_lens=(96)
interp_ratios=(1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0)
# interp_ratios=(2.2 2.4 2.6 2.8 3.0)

bs=32
hd=0
lr=0.001

echo "=== run_weather_multi_pred ==="
echo "Dataset=weather (custom), path=$root_path/$data_path"
echo "seq_len=$seq_len, train_epochs=$train_epochs, stage_num=$stage_num, enc_in=$enc_in"
echo "batch_size=$bs, head_dropout=$hd, lr=$lr"
echo "pred_lens in {${pred_lens[*]}}, interp_ratios in {${interp_ratios[*]}}"

for pl in "${pred_lens[@]}"
do
  for interp_ratio in "${interp_ratios[@]}"
  do
    model_id="LPUNET_${seq_len}_pl${pl}_bs${bs}_d${hd}_lr${lr}_interp${interp_ratio}_LPUNET"

    echo ">>> Running Weather with pred_len=$pl, interp_ratio=$interp_ratio"
    echo ">>> model_id=$model_id"

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
      --head_dropout "$hd" \
      --train_epochs "$train_epochs" \
      --itr "$itr" \
      --batch_size "$bs" \
      --learning_rate "$lr" \
      --interp_ratio "$interp_ratio"

    echo ">>> Finished pred_len=$pl, interp_ratio=$interp_ratio"
  done
done

echo "=== All experiments for Weather finished! ==="