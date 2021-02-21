#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
SERVER_NUM=$3
GPU_NUM_PER_SERVER=$4
MODEL=$5
BACKBONE=$6
BACKBONE_PRETRAINED=$7
BACKBONE_FREEZED=$8
OUTPUT_STRIDE=$9
DISTRIBUTION=${10}
ROUND=${11}
EPOCH=${12}
BATCH_SIZE=${13}
CLIENT_OPTIMIZER=${14}
LR=${15}
DATASET=${16}
DATA_DIR=${17}
EVALUATION_FREQUENCY=${18}
GPU_MAPPING_KEY=${19}
CHECKPOINT_NAME=${20}
CI=${21}

echo $MODEL
echo $BACKBONE
echo $OUTPUT_STRIDE
echo $DATASET
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedseg.py \
  --model $MODEL \
  --backbone $BACKBONE \
  --backbone_pretrained $BACKBONE_PRETRAINED \
  --backbone_freezed $BACKBONE_FREEZED \
  --outstride $OUTPUT_STRIDE \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --checkname CHECKPOINT_NAME \
  --partition_method $DISTRIBUTION \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --batch_size $BATCH_SIZE \
  --client_optimizer $CLIENT_OPTIMIZER \
  --lr $LR \
  --epochs $EPOCH \
  --comm_round $ROUND \
  --evaluation_frequency $EVALUATION_FREQUENCY \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --gpu_mapping_key $GPU_MAPPING_KEY \
  --ci $CI
