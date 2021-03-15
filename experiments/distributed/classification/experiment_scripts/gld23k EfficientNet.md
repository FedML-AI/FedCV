# scigpu
PYTHON=~/anaconda3/envs/py36/bin/python
imagenet_data_dir=/home/datasets/imagenet/ILSVRC2012_dataset
gld_data_dir=~/datasets/landmarks
gld_data_dir=/home/comp/20481896/datasets/landmarks
cifar10_data_dir=~/datasets/cifar10
mnist_data_dir=~/datasets
GPU_UTIL_FILE=scigpu_gpu_util.yaml
MPI_HOST_FILE=scigpu_mpi_host_file

## SGD
### 10 clients


mpirun -np 11 -host scigpu10:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu10:0,4,4,3" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt sgd --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
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
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset gld23k --data_dir /home/esetstore/dataset/gld --partition_method hetero \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1


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
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset gld23k --data_dir /home/esetstore/dataset/gld --partition_method hetero \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.1 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1



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
    --comm_round 4000  --epochs 1 \
    --model efficientnet  \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1


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
    --comm_round 4000  --epochs 1 \
    --model efficientnet  \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.1 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1




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
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset gld23k --data_dir /home/esetstore/dataset/gld --partition_method hetero \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 8000  --epochs 1 \
    --model efficientnet --pretrained \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.1 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999 --sched_fixed 1


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
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset gld23k --data_dir /home/esetstore/dataset/gld --partition_method hetero \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 8000  --epochs 1 \
    --model efficientnet --pretrained \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999


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
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset gld23k --data_dir /home/esetstore/dataset/gld --partition_method hetero \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999


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
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset gld23k --data_dir /home/esetstore/dataset/gld --partition_method hetero \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.3 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999

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
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset gld23k --data_dir /home/esetstore/dataset/gld --partition_method hetero \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999

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
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 100 \
    --dataset gld23k --data_dir /home/esetstore/dataset/gld --partition_method hetero \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 4 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.3 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999


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
    --comm_round 4000  --epochs 1 \
    --model efficientnet --pretrained \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.3 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched StepLR --decay-rounds 1 --decay-rate .999





mpirun -np 11 -host gpu14:1,gpu15:1,gpu16:1,gpu17:1,gpu19:1,gpu20:1,gpu21:1,gpu22:1,gpu23:1,gpu24:1,gpu26:1 \
    -mca btl_tcp_if_include 192.168.0.101/24 \
    ~/miniconda3/bin/python ./main.py \
    --gpu_util_parse "gpu14:1;gpu15:1;gpu16:1;gpu17:1;gpu19:1;gpu20:1;gpu21:1;gpu22:1;gpu23:1;gpu24:1;gpu26:1" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir /nfs_home/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 2000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .9992


mpirun -np 11 -host gpu1:1,gpu3:1,gpu4:1,gpu5:1,gpu6:1,gpu7:1,gpu8:1,gpu9:1,gpu10:1,gpu11:1,gpu13:1 \
    -mca btl_tcp_if_include 192.168.0.101/24 \
    ~/miniconda3/bin/python ./main.py \
    --gpu_util_parse "gpu1:1;gpu3:1;gpu4:1;gpu5:1;gpu6:1;gpu7:1;gpu8:1;gpu9:1;gpu10:1;gpu23:1;gpu24:1;gpu26:1" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir /nfs_home/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 2000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.1 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .9992

mpirun -np 11 -host gpu1:1,gpu3:1,gpu4:1,gpu5:1,gpu6:1,gpu7:1,gpu8:1,gpu9:1,gpu10:1,gpu11:1,gpu13:1 \
    -mca btl_tcp_if_include 192.168.0.101/24 \
    ~/miniconda3/bin/python ./main.py \
    --gpu_util_parse "gpu1:1;gpu3:1;gpu4:1;gpu5:1;gpu6:1;gpu7:1;gpu8:1;gpu9:1;gpu10:1;gpu23:1;gpu24:1;gpu26:1" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir /nfs_home/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 2000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .9992




mpirun -np 5 -host scigpu11:5 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu11:0,0,3,2" \
    --client_num_per_round 5 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir ~/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --comm_round 2000  --epochs 1 \
    --model efficientnet --pretrained \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .9992

mpirun -np 11 -host scigpu11:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu11:4,4,0,3" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir ~/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 2000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .9992

mpirun -np 11 -host scigpu11:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu11:4,4,0,3" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir ~/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 2000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.1 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .9992

mpirun -np 11 -host scigpu11:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu11:4,4,0,3" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir ~/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999

mpirun -np 11 -host scigpu11:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu11:4,4,0,3" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir ~/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.1 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999

mpirun -np 11 -host scigpu11:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu11:4,4,0,3" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir ~/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.3 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999


mpirun -np 11 -host scigpu11:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu11:4,4,3,0" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir ~/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt momentum --lr 0.5 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999


mpirun -np 11 -host scigpu11:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu11:4,4,3,0" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir ~/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt rmsproptf --lr 0.1 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999


mpirun -np 11 -host scigpu11:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu11:4,4,3,0" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir ~/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt rmsproptf --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999

mpirun -np 11 -host scigpu11:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu11:4,4,3,0" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir ~/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt rmsproptf --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999


mpirun -np 11 -host scigpu11:11 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "scigpu11:4,4,3,0" \
    --client_num_per_round 10 --client_num_in_total 233 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset gld23k --data_dir ~/datasets/landmarks \
    --if-timm-dataset -b 32  --data_transform FLTransform \
    --data_load_num_workers 2 \
    --comm_round 4000  --epochs 1 \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt rmsproptf --lr 0.3 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .999
