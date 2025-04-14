#!/bin/bash

# Get first parameter as CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$1
shift

# Count GPU Num (elements separated by comma)
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

echo "[train.sh] Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[train.sh] Detected $NUM_GPUS GPU(s)"

# Get directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/active_mitsuba.sh

# Start training
torchrun --nproc_per_node=$NUM_GPUS train.py "$@"