
lr=2.0
warmup_steps=5000
max_steps=50000
home="/yourhomeaddress/"
savefolder=$home/save/

for spl in logic_question logic_query sql_query sql_question
do
  expname=geo_curriculum_novel_prim_${spl}
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
      --dynamic_curriculum \
      --curriculum_type always_novel_prim \
      --init_training_portion 0.2 \
      --curriculum_ending_time 0.8 \
      --GEO > eval.$i.out 2> eval.$i.err
  done
done
