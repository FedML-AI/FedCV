#!/usr/bin/env bash

NPROC_PER_NODE=$1
NNODE=$2
NODE_RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5
GPU_UTIL=$6
PYTHON=$7
ARGS=$8



CUDA_VISIBLE_DEVICES=$GPU_UTIL $PYTHON -m torch.distributed.launch \
--nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODE --node_rank=$NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
./ddp_classification.py --client_num_in_total $NPROC_PER_NODE $ARGS

# CUDA_VISIBLE_DEVICES=$GPU_UTIL $PYTHON -m torch.distributed.launch \
# --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODE --node_rank=$NODE_RANK \
# --master_addr $MASTER_ADDR \
# --master_port $MASTER_PORT \
# ./ddp_classification.py 
