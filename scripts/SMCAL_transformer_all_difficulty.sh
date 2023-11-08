
lr=2.0
warmup_steps=5000
max_steps=50000
home="/yourhomeaddress/"
savefolder=$home/save/

for difftype in kmeans 
do
  for difflevel in 0 1 2 3 mix0 mix1 mix2
  do
    for split in origin
    do
      for i in `seq 0 4`
      do
        expname=smcal_${split}_${difftype}_difficulty_${difflevel}
        mkdir -p $savefolder/$expname
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
        --smcal_split ${split} \
        --save_memory \
        --special_test_data valid \
        --special_train_data train_${difftype}diff_25_many_${difflevel} \
        --SMCAL > eval.$i.out 2> eval.$i.err
      done
    done
  done
done
