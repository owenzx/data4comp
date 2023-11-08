
lr=2.0
warmup_steps=5000
max_steps=50000
home="/yourhomeaddress/"
savefolder=$home/save/

for spl in logic_query logic_question sql_query sql_question
do
  expname=geo_baseline_${spl}_huge200
  mkdir -p $savefolder/$expname
  for i in `seq 0 4`
  do
      cd $savefolder/$expname
      python -u  $home/main.py \
      --train \
      --model_type transformer \
      --seed $i \
      --n_batch 32 \
      --lr ${lr} \
      --temp 1.0 \
      --beam_size 5 \
      --gclip 5.0 \
      --accum_count 4 \
      --transformer_config 3layer_new \
      --valid_steps 500 \
      --warmup_steps ${warmup_steps} \
      --max_step ${max_steps} \
      --tolarance 50 \
      --exp_name ${expname} \
      --geo_split ${spl} \
      --special_train_data train_m200 \
      --save_memory \
      --GEO > eval.$i.out 2> eval.$i.err
  done
done
