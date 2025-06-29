# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=LPUNET

python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/illness \
    --data_path national_illness.csv \
    --model_id national_illness_$seq_len'_'24 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len 24 \
    --enc_in 7 \
    --des 'Exp' \
    --stage_num 2 \
    --train_epochs 30 \
    --head_dropout 0 \
    --patch_len 8 \
    --itr 1 --batch_size 48 --learning_rate 0.0025  





python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/illness \
    --data_path national_illness.csv \
    --model_id national_illness_$seq_len'_'36 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len 36 \
    --enc_in 7 \
    --des 'Exp' \
    --stage_num 2 \
    --train_epochs 30 \
    --head_dropout 0 \
    --patch_len 4 \
    --itr 1 --batch_size 48 --learning_rate 0.0025   


python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/illness \
    --data_path national_illness.csv \
    --model_id national_illness_$seq_len'_'48 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len 48 \
    --enc_in 7 \
    --des 'Exp' \
    --stage_num 2 \
    --train_epochs 6 \
    --head_dropout 0.15 \
    --patch_len 10 \
    --itr 1 --batch_size 48 --learning_rate 0.004




python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/illness \
    --data_path national_illness.csv \
    --model_id national_illness_$seq_len'_'60 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len 60 \
    --enc_in 7 \
    --des 'Exp' \
    --stage_num 2 \
    --head_dropout 0 \
    --train_epochs 10 \
    --patch_len 8 \
    --itr 1 --batch_size 48 --learning_rate 0.0025 







