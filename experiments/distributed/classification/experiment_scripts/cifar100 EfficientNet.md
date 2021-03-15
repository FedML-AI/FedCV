# scigpu
PYTHON=~/anaconda3/envs/py36/bin/python
imagenet_data_dir=/home/datasets/imagenet/ILSVRC2012_dataset
gld_data_dir=~/datasets/landmarks
gld_data_dir=/home/comp/20481896/datasets/landmarks
cifar10_data_dir=~/datasets/cifar10
cifar100_data_dir=~/datasets/cifar100
mnist_data_dir=~/datasets
GPU_UTIL_FILE=scigpu_gpu_util.yaml
MPI_HOST_FILE=scigpu_mpi_host_file


## SGD
### 10 clients, not pretrain

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 0.1 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999


### 10 clients, pretrain, alpha = 0.1

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 0.1 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained\
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 0.1 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained\
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.003 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 0.1 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained\
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 0.1 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained\
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 0.1 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 0.1 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.003 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 0.1 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 0.1 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1




### 10 clients, pretrain, alpha = 0.5

mpirun  --prefix /home/esetstore/.local/openmpi-4.0.1 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include 192.168.0.1/24 \
    -x NCCL_DEBUG=INFO  \
    -x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
    -x NCCL_IB_DISABLE=1 \
    -bind-to none -map-by slot \
    -np 11 -host  gpu14:5,gpu15:4,gpu13:2 \
    /home/esetstore/pytorch1.4/bin/python ./main.py \
    --gpu_util_parse "gpu14:2,1,1,1;gpu15:1,1,1,1;gpu13:0,0,1,1" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir /home/esetstore/dataset/cifar100 --partition_method hetero --partition_alpha 0.5 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4  \
    --comm_round 4000  --epochs 1 \
    --model mobilenet_v3  --pretrained \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999


mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 0.5 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained\
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 0.5 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained\
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 0.5 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.003 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1


mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 0.5 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 0.5 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 0.5 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 0.5 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1

### 10 clients, pretrain, alpha = 100
mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained\
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.003 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999



mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet  \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.003 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999



mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999




### 10 clients, alpha = 100， pure SGD
mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained\
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt sgd --lr 0.001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt sgd --lr 0.003 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999


mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained\
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt sgd --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt sgd --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999




mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt sgd --lr 0.001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet  \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt sgd --lr 0.003 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999


mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt sgd --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999

mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet  \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt sgd --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999








