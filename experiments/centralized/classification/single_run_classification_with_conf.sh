#!/usr/bin/env bash

GPU_UTIL=$1
DATASET=$2
DATA_DIR=$3
ARGS=$4

source configs/cluster.conf
PYTHON=`cat configs/cluster.conf | grep PYTHON | awk -F= "{print $2}"`
data_dir=`cat configs/cluster.conf | grep $DATA_DIR | awk -F= "{print $2}"`


CUDA_VISIBLE_DEVICES=$GPU_UTIL $PYTHON ./single_classification.py \
    --client_num_in_total 1 \
    --data_dir $data_dir --dataset $DATASET \ 
    $ARGS

# CUDA_VISIBLE_DEVICES=$GPU_UTIL $PYTHON -m torch.distributed.launch \
# --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODE --node_rank=$NODE_RANK \
# --master_addr $MASTER_ADDR \
# --master_port $MASTER_PORT \
# ./ddp_classification.py 
