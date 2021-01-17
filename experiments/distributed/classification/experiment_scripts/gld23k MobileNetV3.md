# ILSVRC2012-100  MobileNetV3-Large-100

# t716
PYTHON=/nfs_home/zhtang/miniconda3/bin/python
imagenet_data_dir=/nfs_home/datasets/ILSVRC2012
gld_data_dir=/nfs_home/datasets/landmarks
cifar10_data_dir=/nfs_home/datasets/cifar10
mnist_data_dir=/nfs_home/datasets/mnist



```
# run on t716
./run_with_conf.sh 10 hostfiles/t716_hostfile_11_2 gld23k gld_data_dir " --gpu_util_file gpuutils/t716_gpu_util.yaml --gpu_util_key 11gpus_2 \
--gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
--model mobilenet_v3 \
--if-timm-dataset -b 16  --data_transform FLTransform --client_num_in_total 233 \
--comm_round 300  --epochs 1 \
--opt rmsproptf --lr 0.03  --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --sched step --decay-rounds 1 --decay-rate .97 \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \ "
```


