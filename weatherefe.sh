# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
# model_name=PatchTST
# model_name=Autoformer
# model_name=Informer
#model_name=DLinear
 model_name=LPUNET


  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/weather \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 21 \
  --des 'Exp' \
  --stage_num 4 \
  --head_dropout 0 \
  --train_epochs 10 \
  --itr 1 --batch_size 128 --learning_rate 0.0005
