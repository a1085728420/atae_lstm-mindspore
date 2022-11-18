#!/bin/bash

if [ $# != 2 ]; then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash convert_dataset.sh DATA_FOLDER GLOVE_FILE"
  echo "for example:"
  echo "  bash convert_dataset.sh \\"
  echo "      /home/workspace/atae_lstm/data \\"
  echo "      /home/workspace/atae_lstm/data/glove.840B.300d.txt"
  echo "It is better to use absolute path."
  echo "=============================================================================================================="
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATA_FOLDER=$(get_real_path $1)
GLOVE_FILE=$(get_real_path $2)
TRAIN_DATA=$DATA_FOLDER/train.mindrecord
EVAL_DATA=$DATA_FOLDER/test.mindrecord

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
echo ${BASE_PATH}

python $BASE_PATH/../create_dataset.py \
    --data_folder=$DATA_FOLDER \
    --glove_file=$GLOVE_FILE \
    --train_data=$TRAIN_DATA \
    --eval_data=$EVAL_DATA
