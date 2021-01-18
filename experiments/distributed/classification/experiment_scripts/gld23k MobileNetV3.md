# ILSVRC2012-100  MobileNetV3-Large-100

# t716
PYTHON=/nfs_home/zhtang/miniconda3/bin/python
imagenet_data_dir=/nfs_home/datasets/ILSVRC2012
gld_data_dir=/nfs_home/datasets/landmarks
cifar10_data_dir=/nfs_home/datasets/cifar10
mnist_data_dir=/nfs_home/datasets/mnist



```
# run on t716
./run_with_conf_t716.sh 10 hostfiles/t716_hostfile_11_2 gld23k gld_data_dir " --gpu_util_file gpuutils/t716_gpu_util.yaml --gpu_util_key 11gpus_2 \
--gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
--model mobilenet_v3 \
--if-timm-dataset -b 16  --data_transform FLTransform --client_num_in_total 233 \
--comm_round 300  --epochs 1 \
--opt rmsproptf --lr 0.03  --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --sched step --decay-rounds 1 --decay-rate .97 \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \ "

./run_with_conf_t716.sh 4 hostfiles/t716_hostfile_5 gld23k gld_data_dir " --gpu_util_file gpuutils/t716_gpu_util.yaml --gpu_util_key 4gpus \
--gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
--model mobilenet_v3 \
--if-timm-dataset -b 16  --data_transform FLTransform --client_num_in_total 233 \
--comm_round 300  --epochs 1 \
--opt rmsproptf --lr 0.03  --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --sched step --decay-rounds 1 --decay-rate .97 \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \ "

# run on scigpu
./run_with_conf.sh 10 hostfiles/scigpu_local_hostfile_11 gld23k gld_data_dir " --gpu_util_file gpuutils/scigpu_gpu_util.yaml --gpu_util_key local_11 \
--gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
--model mobilenet_v3 \
--if-timm-dataset -b 16  --data_transform FLTransform --client_num_in_total 233 \
--comm_round 300  --epochs 1 \
--opt rmsproptf --lr 0.03  --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --sched step --decay-rounds 1 --decay-rate .97 \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \ "


# directly run
# on scigpu

mpirun -np 3 -H scigpu3:2,scigpu4:1 \
    ~/anaconda3/envs/py36/bin/python ./main.py \
    --gpu_util_parse "gpu1:0,0,0,0;gpu1:0,0,0,1;gpu3:0,1,0,0" \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --dataset gld23k --data_dir ~/datasets/landmarks \
    --client_num_per_round 2 --client_num_in_total 233 \
    --if-timm-dataset -b 16  --data_transform FLTransform \
    --comm_round 300  --epochs 1 \
    --model mobilenet_v3 \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt rmsproptf --lr 0.03 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .97 \







# on t716

  -mca btl_tcp_if_include 192.168.0.101/24 \





```


