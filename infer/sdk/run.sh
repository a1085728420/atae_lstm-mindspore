#!/bin/bash

set -e

# Simple log helper functions
info() { echo -e "\033[1;34m[INFO ][MxStream] $1\033[1;37m" ; }
warn() { echo >&2 -e "\033[1;31m[WARN ][MxStream] $1\033[1;37m" ; }

#export MX_SDK_HOME=/home/work/mxVision
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

#to set PYTHONPATH, import the StreamManagerApi.py
export PYTHONPATH=$PYTHONPATH:${MX_SDK_HOME}/python

if [ -d sdk_infer_result ]; then
    rm -rf sdk_infer_result
fi
mkdir sdk_infer_result

python3.7 main.py --pipeline_file=../data/config/atae_lstm.pipeline --data_dir=../data/input --res_dir=sdk_infer_result --do_eval=True &> sdk_infer.log
exit 0