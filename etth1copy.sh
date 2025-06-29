model_name=LPUNET

for pred_len in 96 192 336 512
do
for seq_len in 48 96 192 336 512
do
  
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model LPUNET \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --head_dropout 0.5 \
  --train_epochs 10 \
  --stage_num 4 \
  --itr 1 --batch_size 64 

done
done