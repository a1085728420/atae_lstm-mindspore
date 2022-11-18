#!/bin/bash

if [ $# != 3 ]; then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_eval.sh [DEVICE] [DATA_DIR] [CKPT_FILE]"
  echo "for example:"
  echo "  bash run_eval.sh GPU \\"
  echo "      /home/workspace/atae_lstm/data/ \\"
  echo "      /home/workspace/atae_lstm/train/atae-lstm_max.ckpt"
  echo "It is better to use absolute path."
  echo "=============================================================================================================="
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

DEVICE=$1
DATA_DIR=$(get_real_path "$2")
CKPT_FILE=$(get_real_path "$3")

if [ -d "$BASE_PATH/../eval" ];
then
    rm -rf $BASE_PATH/../eval
fi
mkdir $BASE_PATH/../eval
cd $BASE_PATH/../eval || exit

python $BASE_PATH/../eval.py \
    --eval_ckpt=$CKPT_FILE \
    --device=$DEVICE \
    --data_url=$DATA_DIR > eval_log.log 2>&1 &
