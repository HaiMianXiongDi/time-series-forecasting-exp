model_name=LPUNET

for pred_len in 96 192 336 512
do
for seq_len in 48 96 192 336 512
do
  
  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/weather \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --des 'Exp' \
  --stage_num 3 \
  --head_dropout 0.01 \
  --train_epochs 15 \
  --itr 1 --batch_size 64  

done
done


