#!/usr/bin/env bash

GPU_UTIL=$1
PYTHON=$2
ARGS=$3


CUDA_VISIBLE_DEVICES=$GPU_UTIL $PYTHON ./single_classification.py --client_num_in_total 1 $ARGS

# CUDA_VISIBLE_DEVICES=$GPU_UTIL $PYTHON -m torch.distributed.launch \
# --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODE --node_rank=$NODE_RANK \
# --master_addr $MASTER_ADDR \
# --master_port $MASTER_PORT \
# ./ddp_classification.py 
