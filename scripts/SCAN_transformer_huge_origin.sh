
lr=2.0
warmup_steps=5000
max_steps=50000
home="/yourhomeaddress/"
savefolder=$home/save/

mkdir -p $savefolder/$expname
for i in `seq 0 4`
do
    for pnum in 2 20 200
    do
        expname=scan_around_right_origin_x${pnum}
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
        --scan_split around_right \
        --special_train_data train_primx${pnum} \
        --SCAN_GOODDEV > eval.$i.out 2> eval.$i.err
    done
done
