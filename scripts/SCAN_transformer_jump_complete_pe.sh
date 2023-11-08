
lr=2.0
warmup_steps=5000
max_steps=50000
home="/yourhomeaddress/"
savefolder=$home/save/


for split in hugejump
do
  for pe in 2 8 32 128
  do
    expname=can_${split}_pe${pe}
    mkdir -p $savefolder/$expname
    for i in `seq 0 2`
    do
        cd $savefolder/$expname
        python -u  $home/main.py \
        --train \
        --model_type transformer \
        --seed $i \
        --n_batch 16 \
        --lr ${lr} \
        --temp 1.0 \
        --beam_size 5 \
        --gclip 5.0 \
        --accum_count 8 \
        --transformer_config 3layer_new \
        --valid_steps 500 \
        --warmup_steps ${warmup_steps} \
        --max_step ${max_steps} \
        --tolarance 50 \
        --exp_name ${expname} \
        --scan_split ${split} \
        --save_memory \
        --special_train_data train_hugedatafix_n5000000_l500_p200_pe${pe} \
        --special_dev_data gooddev_addprim_jump_v2 \
        --special_test_data newtest_addprim_jump_v2 \
        --SCAN_GOODDEV > eval.$i.out 2> eval.$i.err
    done
  done
done