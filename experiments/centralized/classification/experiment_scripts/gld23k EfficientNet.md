# ILSVRC2012-100  MobileNetV3-Large-100

# t716
PYTHON=~/miniconda3/bin/python
imagenet_data_dir=/nfs_home/datasets/ILSVRC2012
gld_data_dir=/nfs_home/datasets/landmarks
cifar10_data_dir=/nfs_home/datasets/cifar10
mnist_data_dir=/nfs_home/datasets/mnist



## Pure

```

./single_run_classification.sh "0"  ~/miniconda3/bin/python " --dataset gld23k --data_dir /nfs_home/datasets/landmarks --data_transform FLTransform --model mobilenet_v3 --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.03"
```

## Pure with normal image transform

```

./single_run_classification.sh "0"  ~/miniconda3/bin/python " --dataset gld23k --data_dir /nfs_home/datasets/landmarks --data_transform NormalTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.03"
```


## Add AutoAugmentation
```
./single_run_classification.sh "0"  ~/miniconda3/bin/python " --dataset gld23k --data_dir /nfs_home/datasets/landmarks --data_transform FLTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.03"
```

## Add model EMA
```
./single_run_classification.sh "0"  ~/miniconda3/bin/python " --dataset gld23k --data_dir /nfs_home/datasets/landmarks --data_transform FLTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --remode pixel --reprob 0.2 --lr 0.03"
```


## Add all
```
./single_run_classification.sh "0"  ~/miniconda3/bin/python " --dataset gld23k --data_dir /nfs_home/datasets/landmarks --data_transform NormalTransform --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.03"
```


