# ILSVRC2012-100  MobileNetV3-Large-100

# t716
PYTHON=~/miniconda3/bin/python
imagenet_data_dir=/nfs_home/datasets/ILSVRC2012
gld_data_dir=/nfs_home/datasets/landmarks
cifar10_data_dir=/nfs_home/datasets/cifar10
mnist_data_dir=/nfs_home/datasets/mnist

# DAAI
PYTHON=~/py36/bin/python
imagenet_data_dir=/home/datasets/ILSVRC2012_dataset
gld_data_dir=/home/datasets/landmarks
cifar10_data_dir=/home/datasets/cifar10
GPU_UTIL_FILE=DAAI_gpu_util.yaml
MPI_HOST_FILE=DAAI_mpi_host_file_2


# scigpu
PYTHON=~/anaconda3/envs/py36/bin/python
imagenet_data_dir=/home/datasets/imagenet/ILSVRC2012_dataset
gld_data_dir=/home/comp/20481896/datasets/landmarks
cifar10_data_dir=~/datasets/cifar10
mnist_data_dir=~/datasets
GPU_UTIL_FILE=scigpu_gpu_util.yaml
MPI_HOST_FILE=scigpu_mpi_host_file

## Pure

```

./single_run_classification.sh "0"  ~/anaconda3/envs/py36/bin/python " --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks --data_transform FLTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.03"

./single_run_classification.sh "0"  ~/anaconda3/envs/py36/bin/python " --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks --data_transform FLTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.01"

./single_run_classification.sh "0"  ~/anaconda3/envs/py36/bin/python " --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks --data_transform FLTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.003"
```



## Pure with SGD
./single_run_classification.sh "3" ~/anaconda3/envs/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/comp/20481896/datasets/landmarks --data_transform NormalTransform --model efficientnet --pretrained --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.03"

./single_run_classification.sh "2" ~/anaconda3/envs/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/comp/20481896/datasets/landmarks --data_transform NormalTransform --model efficientnet --pretrained --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.1"

./single_run_classification.sh "2" ~/anaconda3/envs/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/comp/20481896/datasets/landmarks --data_transform NormalTransform --model efficientnet --pretrained --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.3"

./single_run_classification.sh "1" ~/anaconda3/envs/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/comp/20481896/datasets/landmarks --data_transform NormalTransform --model efficientnet --pretrained --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.6"


./single_run_classification.sh "0" ~/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/datasets/landmarks --data_transform NormalTransform --model efficientnet --pretrained --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.03"

./single_run_classification.sh "1" ~/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/datasets/landmarks --data_transform NormalTransform --model efficientnet --pretrained --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.01"

./single_run_classification.sh "2" ~/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/datasets/landmarks --data_transform NormalTransform --model efficientnet --pretrained --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.1"

./single_run_classification.sh "2" ~/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/datasets/landmarks --data_transform NormalTransform --model efficientnet --pretrained --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.3"

./single_run_classification.sh "1" ~/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/datasets/landmarks --data_transform NormalTransform --model efficientnet --pretrained --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.6"

## Pure with normal image transform

```

./single_run_classification.sh "0"  ~/anaconda3/envs/py36/bin/python " --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks --data_transform NormalTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.03"

./single_run_classification.sh "1"  ~/anaconda3/envs/py36/bin/python " --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks --data_transform NormalTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.01"

./single_run_classification.sh "2"  ~/anaconda3/envs/py36/bin/python " --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks --data_transform NormalTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.003"
```


## Add AutoAugmentation
```
./single_run_classification.sh "0"  ~/anaconda3/envs/py36/bin/python " --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks --data_transform FLTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.03"

./single_run_classification.sh "1"  ~/anaconda3/envs/py36/bin/python " --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks --data_transform FLTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.01"

./single_run_classification.sh "3"  ~/anaconda3/envs/py36/bin/python " --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks --data_transform FLTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.003"
```

## Add model EMA
```
# running
./single_run_classification.sh "3"  ~/anaconda3/envs/py36/bin/python " --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks --data_transform FLTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --remode pixel --reprob 0.2 --lr 0.03"

./single_run_classification.sh "1"  ~/anaconda3/envs/py36/bin/python " --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks --data_transform FLTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --remode pixel --reprob 0.2 --lr 0.01"

./single_run_classification.sh "0"  ~/anaconda3/envs/py36/bin/python " --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks --data_transform FLTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --remode pixel --reprob 0.2 --lr 0.003"

```


## Add all
```
./single_run_classification.sh "2"  ~/anaconda3/envs/py36/bin/python " --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks --data_transform NormalTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.03"

./single_run_classification.sh "3"  ~/anaconda3/envs/py36/bin/python " --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks --data_transform NormalTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.01"

./single_run_classification.sh "2"  ~/anaconda3/envs/py36/bin/python " --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks --data_transform NormalTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.003"
```

### SGD with all tricks
./single_run_classification.sh "3" ~/anaconda3/envs/py36/bin/python " --dataset gld23k --data_dir /home/comp/20481896/datasets/landmarks --data_transform NormalTransform --model efficientnet --pretrained --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.01"


./single_run_classification.sh "3" ~/py36/bin/python " --dataset gld23k --data_dir /home/datasets/landmarks --data_transform NormalTransform --model efficientnet --pretrained --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.01"

./single_run_classification.sh "3" ~/py36/bin/python " --dataset gld23k --data_dir /home/datasets/landmarks --data_transform NormalTransform --model efficientnet --pretrained --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.03"

./single_run_classification.sh "1" ~/py36/bin/python " --dataset gld23k --data_dir /home/datasets/landmarks --data_transform NormalTransform --model efficientnet --pretrained --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.1"

./single_run_classification.sh "2" ~/py36/bin/python " --dataset gld23k --data_dir /home/datasets/landmarks --data_transform NormalTransform --model efficientnet --pretrained --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.3"

./single_run_classification.sh "3" ~/py36/bin/python " --dataset gld23k --data_dir /home/datasets/landmarks --data_transform NormalTransform --model efficientnet --pretrained --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.6"


