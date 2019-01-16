#!/usr/bin/env bash
cd ../
batch_size=64
maxlen=1024
epochs=100
lr=0.01
lr_halve_interval=3
gamma=0.9
snapshot_interval=1
gpuid=0
nthreads=4
 
dataset="ag_news"
data_folder="datasets/${dataset}/svdcnn"
depth=9
model_folder="models/${dataset}_${depth}"
shortcut=True

python -m src.main --dataset ${dataset} \
                         --model_folder ${model_folder} \
                         --data_folder ${data_folder} \
                         --depth ${depth} \
                         --maxlen ${maxlen} \
                         --batch_size ${batch_size} \
                         --epochs ${epochs} \
                         --lr ${lr} \
                         --lr_halve_interval ${lr_halve_interval} \
                         --snapshot_interval ${snapshot_interval} \
                         --gamma ${gamma} \
                         --gpuid ${gpuid} \
                         --nthreads ${nthreads} \
                         --shortcut ${shortcut} \