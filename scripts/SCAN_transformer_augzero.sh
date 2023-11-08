
lr=2.0
warmup_steps=5000
max_steps=50000
home="/yourhomeaddress/"
savefolder=$home/save/

expname=scan_around_right_difficulty_len
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
    --scan_split around_right \
    --special_train_data trainm200 \
    --save_memory \
    --SCAN_GOODDEV > eval.$i.out 2> eval.$i.err
done
