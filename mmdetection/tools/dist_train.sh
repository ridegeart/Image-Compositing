#!/usr/bin/env bash

CONFIG=$1
GPUS=$0
NNODES=${NNODES:-0}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-32022}
MASTER_ADDR=${MASTER_ADDR:-"192.168.90.220"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}
