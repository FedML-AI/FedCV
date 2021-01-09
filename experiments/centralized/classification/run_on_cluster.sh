#!/usr/bin/env bash

GPU_UTIL=$1
CLUSTER=$2
DATASET=$3
ARGS=$4

if [ $CLUSTER = "scigpu" ]; then
    echo "Running on scigpu"
    PYTHON=~/anaconda3/envs/py36/bin/python
    imagenet_data_dir=/home/datasets/imagenet/ILSVRC2012_dataset
    gld_data_dir=~/datasets/landmarks
    cifar10_data_dir=~/datasets/cifar10
elif [ $CLUSTER = "DAAI" ] then
    echo "Running on DAAI"
    PYTHON=~/py36/bin/python
    imagenet_data_dir=/home/datasets/ILSVRC2012_dataset
    gld_data_dir=/home/datasets/landmarks
    cifar10_data_dir=/home/datasets/cifar10
elif [ $CLUSTER = "gpuhome" ] then
    echo "Running on gpuhome"
    PYTHON=~/py36/bin/python
    cifar10_data_dir=/home/comp/zhtang/dc2-p2p-dl2/data
    mnist_data_dir=/home/comp/zhtang/dc2-p2p-dl2/data
elif [ $CLUSTER = "t716" ] then
    echo "Running on t716"
    PYTHON=~/miniconda3/bin/python
    imagenet_data_dir=/nfs_home/datasets/ILSVRC2012
    cifar10_data_dir=/nfs_home/datasets/cifar10
    mnist_data_dir=/nfs_home/datasets/mnist
elif [ $CLUSTER = "esetstore" ] then
    echo "Running on esetstore"
    PYTHON=""
    imagenet_data_dir=""
    cifar10_data_dir=""
    mnist_data_dir=""
elif [ $CLUSTER = "csr" ] then
    echo "Running on csr"
    PYTHON=~/miniconda3/bin/python
    cifar10_data_dir=/home/comp/zhtang/datasets/cifar10
    mnist_data_dir=/home/comp/zhtang/datasets/mnist
else
    echo "No this cluster"
fi

if [ $DATASET = "mnist" ]; then
    echo "Using dataset mnist"
elif [ $DATASET = "cifar10" ]; then
    echo "Using dataset cifar10"
elif [ $DATASET = "ILSVRC2012-100" ]; then
    echo "Using dataset ILSVRC2012-100"
elif [ $DATASET = "ILSVRC2012" ]; then
    echo "Using dataset ILSVRC2012"
else:
    echo "No this dataest"
fi


CUDA_VISIBLE_DEVICES=$GPU_UTIL $PYTHON ./single_classification.py --client_num_in_total 1 $ARGS

# CUDA_VISIBLE_DEVICES=$GPU_UTIL $PYTHON -m torch.distributed.launch \
# --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODE --node_rank=$NODE_RANK \
# --master_addr $MASTER_ADDR \
# --master_port $MASTER_PORT \
# ./ddp_classification.py 



