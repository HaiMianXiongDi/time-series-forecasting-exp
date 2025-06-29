model_name=LPUNET

for pred_len in 512
do
for seq_len in 48 96 192 336 512
do

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/electricity \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 321 \
  --des 'Exp' \
  --stage_num 4 \
  --head_dropout 0.01 \
  --train_epochs 10 \
  --patch_len 16 \
  --itr 1 --batch_size 32 --learning_rate 0.0005 >logs/LongForecasting/electricitytest_$seq_len'_'$pred_len.log

done
done









