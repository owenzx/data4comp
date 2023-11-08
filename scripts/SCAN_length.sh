
lr=2.0
warmup_steps=5000
max_steps=50000
home="/yourhomeaddress/"
savefolder=$home/save/


for l in 62 125 250 500
do
  for split in hugelength
  do
    expname=scan_${split}_l${l}
    mkdir -p $savefolder/$expname
    for i in `seq 0 4`
    do
        cd $savefolder/$expname
        python -u  $home/main.py \
        --train \
        --model_type transformer \
        --seed $i \
        --n_batch 128 \
        --lr ${lr} \
        --temp 1.0 \
        --beam_size 5 \
        --gclip 5.0 \
        --accum_count 1 \
        --transformer_config 3layer_new \
        --valid_steps 500 \
        --warmup_steps ${warmup_steps} \
        --max_step ${max_steps} \
        --tolarance 50 \
        --exp_name ${expname} \
        --scan_split ${split} \
        --special_train_data train_hugev3_l${l} \
        --special_dev_data dev_small_hugev3_l${l} \
        --special_test_data test_small_hugev3_l${l} \
        --SCAN_GOODDEV > eval.$i.out 2> eval.$i.err
    done
  done
done
