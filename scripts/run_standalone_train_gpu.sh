#!/bin/bash

if [ $# != 1 ]; then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_standalone_train_gpu.sh DATA_DIR"
  echo "for example:"
  echo "  bash run_standalone_train_gpu.sh \\"
  echo "      /home/workspace/atae_lstm/data/"
  echo "It is better to use absolute path."
  echo "=============================================================================================================="
  exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    realpath -m "$PWD"/"$1"
  fi
}

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
echo ${BASE_PATH}

ulimit -u unlimited
TRAIN_DIR=$(get_real_path "$1")

if [ -d "$BASE_PATH/../train" ];
then
    rm -rf $BASE_PATH/../train
fi
mkdir $BASE_PATH/../train
cd $BASE_PATH/../train || exit

echo "start standalone training on GPU."

python -u $BASE_PATH/../train.py \
    --data_url=$TRAIN_DIR \
    --parallel=False > net_log.log 2>&1 &
