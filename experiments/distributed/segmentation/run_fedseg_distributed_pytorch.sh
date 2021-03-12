#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
MODEL=$3
BACKBONE=$4
BACKBONE_PRETRAINED=$5
OUTPUT_STRIDE=$6
IMAGE_SIZE=$7
DISTRIBUTION=$8
ROUND=$9
EPOCH=${10}
BATCH_SIZE=${11}
CLIENT_OPTIMIZER=${12}
LR=${13}
DATASET=${14}
DATA_DIR=${15}
EVALUATION_FREQUENCY=${16}
GPU_MAPPING_KEY=${17}
CHECKPOINT_NAME=${18}
PROCESS_NAME=${19}
CI=${20}

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
  --outstride $OUTPUT_STRIDE \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --image_size $IMAGE_SIZE \
  --checkname $CHECKPOINT_NAME \
  --partition_method $DISTRIBUTION \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --batch_size $BATCH_SIZE \
  --client_optimizer $CLIENT_OPTIMIZER \
  --lr $LR \
  --epochs $EPOCH \
  --comm_round $ROUND \
  --evaluation_frequency $EVALUATION_FREQUENCY \
  --gpu_mapping_key $GPU_MAPPING_KEY \
  --process_name $PROCESS_NAME \
  --ci $CI
