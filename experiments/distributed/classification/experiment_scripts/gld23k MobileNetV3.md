# ILSVRC2012-100  MobileNetV3-Large-100

# t716
PYTHON=/nfs_home/zhtang/miniconda3/bin/python
imagenet_data_dir=/nfs_home/datasets/ILSVRC2012
gld_data_dir=/nfs_home/datasets/landmarks
cifar10_data_dir=/nfs_home/datasets/cifar10
mnist_data_dir=/nfs_home/datasets/mnist


## SGD
### 10 clients
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
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset gld23k --data_dir /home/esetstore/dataset/gld --partition_method hetero \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 8000  --epochs 1 \
    --model mobilenet_v3 --pretrained \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.1 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999



```
# directly run
# on scigpu

mpirun -np 3 -host scigpu10:2,scigpu13:1 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:1,1,0,0;scigpu13:0,0,1,0" \
    --client_num_per_round 2 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir ~/datasets/landmarks \
    --if-timm-dataset -b 16  --data_transform FLTransform \
    --comm_round 300  --epochs 1 \
    --model mobilenet_v3  --pretrained \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt rmsproptf --lr 0.03 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .97



# on t716


mpirun -np 3 -host gpu1:2,gpu3:1 \
    -mca btl_tcp_if_include 192.168.0.101/24 \
    ~/miniconda3/bin/python ./main.py \
    --client_num_per_round 2 --client_num_in_total 233 \
    --gpu_util_parse "gpu1:2;gpu3:1" \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir /nfs_home/datasets/landmarks \
    --data_load_num_workers 2 \
    --if-timm-dataset -b 16  --data_transform FLTransform \
    --comm_round 300  --epochs 1 \
    --model mobilenet_v3 \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt rmsproptf --lr 0.03 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .97


```

# 5 clients
```
mpirun -np 6 -host gpu1:1,gpu3:1,gpu4:1,gpu5:1,gpu6:1,gpu7:1 \
    -mca btl_tcp_if_include 192.168.0.101/24 \
    ~/miniconda3/bin/python ./main.py \
    --gpu_util_parse "gpu1:1;gpu3:1;gpu4:1;gpu5:1;gpu6:1;gpu7:1" \
    --client_num_per_round 5 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir /nfs_home/datasets/landmarks \
    --data_load_num_workers 2 \
    --if-timm-dataset -b 16  --data_transform FLTransform \
    --comm_round 300  --epochs 1 \
    --model mobilenet_v3 \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt rmsproptf --lr 0.01 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .97
```


# 10 clients
```
# bad, killed
mpirun -np 11 -host gpu1:1,gpu3:1,gpu4:1,gpu5:1,gpu6:1,gpu7:1,gpu8:1,gpu9:1,gpu10:1,gpu11:1,gpu13:1 \
    -mca btl_tcp_if_include 192.168.0.101/24 \
    ~/miniconda3/bin/python ./main.py \
    --gpu_util_parse "gpu1:1;gpu3:1;gpu4:1;gpu5:1;gpu6:1;gpu7:1;gpu8:1;gpu9:1;gpu10:1;gpu11:1;gpu13:1" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir /nfs_home/datasets/landmarks \
    --if-timm-dataset -b 16  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 1000  --epochs 1 \
    --model mobilenet_v3 \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt rmsproptf --lr 0.01 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .97

mpirun -np 11 -host gpu1:1,gpu27:1,gpu4:1,gpu5:1,gpu6:1,gpu7:1,gpu8:1,gpu9:1,gpu10:1,gpu11:1,gpu13:1 \
    -mca btl_tcp_if_include 192.168.0.101/24 \
    ~/miniconda3/bin/python ./main.py \
    --gpu_util_parse "gpu1:1;gpu27:1;gpu4:1;gpu5:1;gpu6:1;gpu7:1;gpu8:1;gpu9:1;gpu10:1;gpu11:1;gpu13:1" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir /nfs_home/datasets/landmarks \
    --if-timm-dataset -b 16  --data_transform FLTransform \
    --comm_round 1000  --epochs 1 \
    --model mobilenet_v3 \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt rmsproptf --lr 0.01 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .97

mpirun -np 11 -host gpu1:1,gpu3:1,gpu4:1,gpu5:1,gpu6:1,gpu7:1,gpu8:1,gpu9:1,gpu10:1,gpu11:1,gpu13:1 \
    -mca btl_tcp_if_include 192.168.0.101/24 \
    ~/miniconda3/bin/python ./main.py \
    --gpu_util_parse "gpu1:1;gpu3:1;gpu4:1;gpu5:1;gpu6:1;gpu7:1;gpu8:1;gpu9:1;gpu10:1;gpu11:1;gpu13:1" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir /nfs_home/datasets/landmarks \
    --if-timm-dataset -b 16  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 1000  --epochs 1 \
    --model mobilenet_v3 \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt rmsproptf --lr 0.001 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .992

mpirun -np 11 -host gpu14:1,gpu15:1,gpu16:1,gpu17:1,gpu19:1,gpu20:1,gpu21:1,gpu22:1,gpu23:1,gpu24:1,gpu26:1 \
    -mca btl_tcp_if_include 192.168.0.101/24 \
    ~/miniconda3/bin/python ./main.py \
    --gpu_util_parse "gpu14:1;gpu15:1;gpu16:1;gpu17:1;gpu19:1;gpu20:1;gpu21:1;gpu22:1;gpu23:1;gpu24:1;gpu26:1" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir /nfs_home/datasets/landmarks \
    --if-timm-dataset -b 64  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 1000  --epochs 1 \
    --model mobilenet_v3 \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt rmsproptf --lr 0.003 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .992
```

## SGD
### 10 clients
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
    --dataset gld23k --data_dir /home/esetstore/dataset/gld --partition_method hetero \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 10000  --epochs 1 \
    --model mobilenet_v3 --pretrained \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.1 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999




mpirun -np 11 -host gpu1:1,gpu3:1,gpu4:1,gpu5:1,gpu6:1,gpu7:1,gpu8:1,gpu9:1,gpu10:1,gpu11:1,gpu13:1 \
    -mca btl_tcp_if_include 192.168.0.101/24 \
    ~/miniconda3/bin/python ./main.py \
    --gpu_util_parse "gpu1:1;gpu3:1;gpu4:1;gpu5:1;gpu6:1;gpu7:1;gpu8:1;gpu9:1;gpu10:1;gpu11:1;gpu13:1" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir /nfs_home/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 2000  --epochs 1 \
    --model mobilenet_v3 \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .992


mpirun -np 11 -host gpu1:1,gpu3:1,gpu4:1,gpu5:1,gpu6:1,gpu7:1,gpu8:1,gpu9:1,gpu10:1,gpu11:1,gpu13:1 \
    -mca btl_tcp_if_include 192.168.0.101/24 \
    ~/miniconda3/bin/python ./main.py \
    --gpu_util_parse "gpu1:1;gpu3:1;gpu4:1;gpu5:1;gpu6:1;gpu7:1;gpu8:1;gpu9:1;gpu10:1;gpu11:1;gpu13:1" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir /nfs_home/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 2000  --epochs 1 \
    --model mobilenet_v3 \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.1 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .992

mpirun -np 11 -host gpu1:1,gpu3:1,gpu4:1,gpu5:1,gpu6:1,gpu7:1,gpu8:1,gpu9:1,gpu10:1,gpu11:1,gpu13:1 \
    -mca btl_tcp_if_include 192.168.0.101/24 \
    ~/miniconda3/bin/python ./main.py \
    --gpu_util_parse "gpu1:1;gpu3:1;gpu4:1;gpu5:1;gpu6:1;gpu7:1;gpu8:1;gpu9:1;gpu10:1;gpu11:1;gpu13:1" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir /nfs_home/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 2000  --epochs 1 \
    --model mobilenet_v3 \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .992


## SGD with aa and model ema












