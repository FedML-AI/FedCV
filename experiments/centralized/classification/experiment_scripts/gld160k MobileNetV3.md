# ILSVRC2012-100  MobileNetV3-Large-100

# scigpu
PYTHON=~/anaconda3/envs/py36/bin/python
imagenet_data_dir=/home/datasets/imagenet/ILSVRC2012_dataset
gld_data_dir=/home/comp/20481896/datasets/landmarks
gld_data_dir=~/datasets/landmarks
cifar10_data_dir=/home/comp/20481896/datasets/cifar10
cifar10_data_dir=~/datasets/cifar10
GPU_UTIL_FILE=scigpu_gpu_util.yaml
MPI_HOST_FILE=scigpu_mpi_host_file

# DAAI
PYTHON=~/py36/bin/python
imagenet_data_dir=/home/datasets/ILSVRC2012_dataset
gld_data_dir=/home/datasets/landmarks
cifar10_data_dir=/home/datasets/cifar10
GPU_UTIL_FILE=DAAI_gpu_util.yaml
MPI_HOST_FILE=DAAI_mpi_host_file_2

# t716
PYTHON=~/miniconda3/bin/python
imagenet_data_dir=/nfs_home/datasets/ILSVRC2012
gld_data_dir=/nfs_home/datasets/landmarks
cifar10_data_dir=/nfs_home/datasets/cifar10
mnist_data_dir=/nfs_home/datasets/mnist


## Pure

```

./single_run_classification.sh "0"  ~/anaconda3/envs/py36/bin/python " --dataset gld160k --data_dir /home/comp/20481896/datasets/landmarks --data_transform FLTransform --model mobilenet_v3 --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.03"
```

## Pure with normal image transform

```

./single_run_classification.sh "1"  ~/anaconda3/envs/py36/bin/python " --dataset gld160k --data_dir /home/comp/20481896/datasets/landmarks --data_transform NormalTransform --model mobilenet_v3 --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.03"
```


## Add AutoAugmentation
```
./single_run_classification.sh "2"  ~/anaconda3/envs/py36/bin/python " --dataset gld160k --data_dir /home/comp/20481896/datasets/landmarks --data_transform FLTransform --model mobilenet_v3 --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.03"
```

## Add model EMA
```
./single_run_classification.sh "0"  ~/anaconda3/envs/py36/bin/python " --dataset gld160k --data_dir /home/comp/20481896/datasets/landmarks --data_transform FLTransform --model mobilenet_v3 --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --remode pixel --reprob 0.2 --lr 0.03"
```


## Add all
```
./single_run_classification.sh "0"  ~/anaconda3/envs/py36/bin/python " --dataset gld160k --data_dir /home/comp/20481896/datasets/landmarks --data_transform NormalTransform --model mobilenet_v3 --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.03"
```


