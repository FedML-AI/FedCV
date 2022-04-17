#!/usr/bin/env bash

WORKER_NUM=$1
MPI_HOST_FILE=$2
DATASET=$3
DATA_DIR=$4
ARGS=$5


source configs/cluster.conf
PYTHON=`cat configs/cluster.conf | grep PYTHON | awk -F "=" '{print $2}'`
data_dir=`cat configs/cluster.conf | grep $DATA_DIR | awk -F "=" '{print $2}'`


PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM
echo $MPI_HOST_FILE
echo $PYTHON
echo $data_dir




mpirun -np $PROCESS_NUM -hostfile ./$MPI_HOST_FILE \
  $PYTHON ./main.py \
  --data_dir $data_dir --dataset $DATASET \
  --client_num_per_round $WORKER_NUM \
  $ARGS


