#!/usr/bin/env bash

gpu=0
log_folder="dbp15k_log"
model="AliNet"

while getopts "g:l:m:" opt;
do
    case ${opt} in
        g) gpu=$OPTARG ;;
        l) log_folder=$OPTARG ;;
        m) model=$OPTARG ;;
    esac
done

echo "log folder: " "${log_folder}"
if [[ ! -d ${log_folder} ]];then
    mkdir -p "${log_folder}"
    echo "create log folder: " "${log_folder}"
fi

neg_multi=(10)
neg_margin=(1.5)
neg_param=(0.1)
epsilon=(0.98)
batch_size=(4500)
sim_th=(0.5)

prefix='../data/DBP15K/'
suffix='/mtranse/0_3/'
inputs=('zh_en' 'ja_en' 'fr_en')

for bs in "${batch_size[@]}"
do
for nmu in "${neg_multi[@]}"
do
for nma in "${neg_margin[@]}"
do
for npa in "${neg_param[@]}"
do
for ep in "${epsilon[@]}"
do
for st in "${sim_th[@]}"
do
for in in "${inputs[@]}"
do
path=${prefix}${in}${suffix}
echo "input: " "${path}"
cur_time="$(date +%Y%m%d%H%M%S)"
CUDA_VISIBLE_DEVICES=${gpu} python main.py \
                        --embedding_module "${model}" \
                        --input "${path}" \
                        --neg_multi "${nmu}" \
                        --neg_margin "${nma}" \
                        --neg_param "${npa}" \
                        --truncated_epsilon "${ep}" \
                        --batch_size "${bs}" \
                        --sim_th "${st}" \
                        > "${log_folder}"/"${model}"_"${in}"_"${nmu}"_"${nma}"_"${npa}"_"${ep}"_"${bs}"_"${st}"_${cur_time}
sleep 10
done
done
done
done
done
done
done