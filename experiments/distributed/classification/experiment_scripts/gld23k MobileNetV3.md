# ILSVRC2012-100  MobileNetV3-Large-100

# t716
PYTHON=/nfs_home/zhtang/miniconda3/bin/python
imagenet_data_dir=/nfs_home/datasets/ILSVRC2012
gld_data_dir=/nfs_home/datasets/landmarks
cifar10_data_dir=/nfs_home/datasets/cifar10
mnist_data_dir=/nfs_home/datasets/mnist



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
    --model mobilenet_v3 \
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












