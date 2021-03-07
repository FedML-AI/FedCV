# ILSVRC2012-100  visTransfomer

# t716
PYTHON=/home/esetstore/pytorch1.4/bin/python
imagenet_data_dir=/nfs_home/datasets/ILSVRC2012
gld_data_dir=/nfs_home/datasets/landmarks
cifar10_data_dir=/nfs_home/datasets/cifar10
mnist_data_dir=/nfs_home/datasets/mnist

# esetstore
PYTHON=/home/esetstore/pytorch1.4/bin/python
imagenet_data_dir=/home/esetstore/dataset/ILSVRC2012_dataset
gld_data_dir=/home/esetstore/dataset/gld


# directly run
# on scigpu

## SGD
### 10 clients, 1000 total, not dirichlet
cd ~/zhtang/FedCV/experiments/distributed/classification
mpirun  --prefix /home/esetstore/.local/openmpi-4.0.1 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include 192.168.0.1/24 \
    -x NCCL_DEBUG=INFO  \
    -x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
    -x NCCL_IB_DISABLE=1 \
    -bind-to none -map-by slot \
    -np 11 -host  gpu9:5,gpu10:4,gpu8:2 \
    /home/esetstore/pytorch1.4/bin/python ./main.py \
    --gpu_util_parse "gpu9:2,1,1,1;gpu10:1,1,1,1;gpu8:0,0,1,1" \
    --client_num_per_round 10 --client_num_in_total 1000 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 400 \
    --dataset ILSVRC2012-100 --data_dir /home/esetstore/dataset/ILSVRC2012_dataset --partition_method hetero \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 8000  --epochs 1 \
    --model efficientnet --pretrained \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.1 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999



cd ~/zhtang/FedCV/experiments/distributed/classification
mpirun  --prefix /home/esetstore/.local/openmpi-4.0.1 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include 192.168.0.1/24 \
    -x NCCL_DEBUG=INFO  \
    -x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
    -x NCCL_IB_DISABLE=1 \
    -bind-to none -map-by slot \
    -np 11 -host  gpu1:5,gpu2:4,gpu3:2 \
    /home/esetstore/pytorch1.4/bin/python ./main.py \
    --gpu_util_parse "gpu1:2,1,1,1;gpu2:1,1,1,1;gpu3:1,1,0,0" \
    --client_num_per_round 10 --client_num_in_total 1000 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset ILSVRC2012-100 --data_dir /home/esetstore/dataset/ILSVRC2012_dataset \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --opt sgd --momentum 0 --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999


cd ~/zhtang/FedCV/experiments/distributed/classification
mpirun  --prefix /home/esetstore/.local/openmpi-4.0.1 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include 192.168.0.1/24 \
    -x NCCL_DEBUG=INFO  \
    -x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
    -x NCCL_IB_DISABLE=1 \
    -bind-to none -map-by slot \
    -np 11 -host  gpu4:5,gpu5:4,gpu3:2 \
    /home/esetstore/pytorch1.4/bin/python ./main.py \
    --gpu_util_parse "gpu4:2,1,1,1;gpu5:1,1,1,1;gpu3:0,0,1,1" \
    --client_num_per_round 10 --client_num_in_total 1000 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset ILSVRC2012-100 --data_dir /home/esetstore/dataset/ILSVRC2012_dataset \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --opt sgd --momentum 0  --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999



## SGD with momentum
### 10 clients, 1000 total, not dirichlet
```

cd ~/zhtang/FedCV/experiments/distributed/classification
mpirun  --prefix /home/esetstore/.local/openmpi-4.0.1 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include 192.168.0.1/24 \
    -x NCCL_DEBUG=INFO  \
    -x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
    -x NCCL_IB_DISABLE=1 \
    -bind-to none -map-by slot \
    -np 11 -host  gpu9:5,gpu10:4,gpu8:2 \
    /home/esetstore/pytorch1.4/bin/python ./main.py \
    --gpu_util_parse "gpu9:2,1,1,1;gpu10:1,1,1,1;gpu8:0,0,1,1" \
    --client_num_per_round 10 --client_num_in_total 1000 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset ILSVRC2012-100 --data_dir /home/esetstore/dataset/ILSVRC2012_dataset \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --opt momentum --lr 0.1 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999


cd ~/zhtang/FedCV/experiments/distributed/classification
mpirun  --prefix /home/esetstore/.local/openmpi-4.0.1 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include 192.168.0.1/24 \
    -x NCCL_DEBUG=INFO  \
    -x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
    -x NCCL_IB_DISABLE=1 \
    -bind-to none -map-by slot \
    -np 11 -host  gpu4:5,gpu5:4,gpu3:2 \
    /home/esetstore/pytorch1.4/bin/python ./main.py \
    --gpu_util_parse "gpu4:2,1,1,1;gpu5:1,1,1,1;gpu3:0,0,1,1" \
    --client_num_per_round 10 --client_num_in_total 1000 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset ILSVRC2012-100 --data_dir /home/esetstore/dataset/ILSVRC2012_dataset \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --opt momentum --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999








mpirun -np 11 -host  gpu5:10 \
    /home/esetstore/pytorch1.4/bin/python ./main.py \
    --gpu_util_parse "gpu5:3,3,3,2" \
    --client_num_per_round 10 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset gld23k --data_dir ~/datasets/landmarks \
    --partition_alpha 1.0 \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --opt momentum --lr 0.1 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999







