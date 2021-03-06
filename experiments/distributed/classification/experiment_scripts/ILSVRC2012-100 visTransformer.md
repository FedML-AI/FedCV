# ILSVRC2012-100  visTransfomer

# t716
PYTHON=/nfs_home/zhtang/miniconda3/bin/python
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
```
mpirun -np 3 -host scigpu11:3 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu11:2,1,0,0" \
    --client_num_per_round 2 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset ILSVRC2012-100 --data_dir /home/datasets/imagenet/ILSVRC2012_dataset \
    --if-timm-dataset -b 16  --data_transform FLTransform \
    --comm_round 300  --epochs 1 \
    --model visTransformer \
    --opt rmsproptf --lr 0.03 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .97
```
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
    -np 11 -host  gpu11:5,gpu12:4,gpu13:2 \
    /home/esetstore/pytorch1.4/bin/python ./main.py \
    --gpu_util_parse "gpu11:2,1,1,1;gpu12:1,1,1,1;gpu13:1,1,0,0" \
    --client_num_per_round 10 --client_num_in_total 1000 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset ILSVRC2012-100 --data_dir /home/esetstore/dataset/ILSVRC2012_dataset \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 4000  --epochs 1 \
    --model visTransformer \
    --opt sgd --momentum 0  --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999


cd ~/zhtang/FedCV/experiments/distributed/classification
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
    --client_num_per_round 10 --client_num_in_total 1000 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset ILSVRC2012-100 --data_dir /home/esetstore/dataset/ILSVRC2012_dataset \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 4000  --epochs 1 \
    --model visTransformer \
    --opt sgd --momentum 0  --lr 0.001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999












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
    -np 11 -host  gpu11:5,gpu12:4,gpu13:2 \
    /home/esetstore/pytorch1.4/bin/python ./main.py \
    --gpu_util_parse "gpu11:2,1,1,1;gpu12:1,1,1,1;gpu13:1,1,0,0" \
    --client_num_per_round 10 --client_num_in_total 1000 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset ILSVRC2012-100 --data_dir /home/esetstore/dataset/ILSVRC2012_dataset \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 4000  --epochs 1 \
    --model visTransformer \
    --opt momentum --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999


cd ~/zhtang/FedCV/experiments/distributed/classification
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
    --client_num_per_round 10 --client_num_in_total 1000 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset ILSVRC2012-100 --data_dir /home/esetstore/dataset/ILSVRC2012_dataset \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 4000  --epochs 1 \
    --model visTransformer \
    --opt momentum --lr 0.001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999



