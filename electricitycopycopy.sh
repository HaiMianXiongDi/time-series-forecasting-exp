# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=336
model_name=LPUNET

for stage_num in {1..6}
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/electricity \
      --data_path electricity.csv \
      --model_id Electricity_${seq_len}'_'96_stage${stage_num} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --pred_len 96 \
      --enc_in 321 \
      --des 'Exp' \
      --stage_num $stage_num \
      --head_dropout 0.01 \
      --train_epochs 30 \
      --patch_len 16 \
      --itr 1 --batch_size 32 --learning_rate 0.0005 >logs/LongForecasting/electricity_${seq_len}'_'96_stage${stage_num}.log
done
