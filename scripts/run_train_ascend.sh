#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_train_ascend.sh DATA_DIR OUTPUT_DIR"
echo "for example:"
echo "sh run_train_ascend.sh \
        /home/workspace/atae_lstm/data/ \
        /home/workspace/atae_lstm/train/"
echo "It is better to use absolute path."
echo "=============================================================================================================="

TRAIN_DIR=$1
OUTPUT_DIR=$2

export GLOG_v=2

current_exec_path=$(pwd)
echo ${current_exec_path}
if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train

echo "start for training"

python train.py \
    --config=${current_exec_path}/src/model_utils/config.json \
    --data_url=$TRAIN_DIR \
    --train_url=$OUTPUT_DIR > ./train/net_log.log 2>&1 &
