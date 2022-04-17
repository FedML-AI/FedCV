
# PyTorch DDP classification

## lr_scheduler parameter reference:


EfficientNet-B0 with RandAugment - 77.7 top-1, 95.3 top-5



```

sh run_classification.sh 8 1 0 127.0.0.1 11111 "0,3"  ~/anaconda3/envs/py36/bin/python " --dataset ILSVRC2012 --data_dir /home/datasets/imagenet/ILSVRC2012_dataset  --model efficientnet -b 384 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.048"

# crashed need to rerun
sh run_classification.sh 3 1 0 127.0.0.1 11112 "0,2,3"  ~/anaconda3/envs/py36/bin/python " --dataset ILSVRC2012 --data_dir /home/datasets/imagenet/ILSVRC2012_dataset --model efficientnet --distributed --if-timm-dataset -b 256 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .048"

sh run_classification.sh 3 1 0 127.0.0.1 11112 "0,2,3"  ~/anaconda3/envs/py36/bin/python " --dataset ILSVRC2012 --data_dir /home/datasets/imagenet/ILSVRC2012_dataset --model efficientnet --distributed --if-timm-dataset -b 256 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .048"


sh run_classification.sh 3 1 0 127.0.0.1 11112 "0,1,2"  ~/anaconda3/envs/py36/bin/python " --dataset ILSVRC2012 --data_dir /home/datasets/imagenet/ILSVRC2012_dataset --model efficientnet --distributed --if-timm-dataset -b 256 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .048"

```


MobileNetV3-Large-100 - 75.766 top-1, 92,542 top-5



```
# crashed need to rerun
sh run_classification.sh 4 1 0 127.0.0.1 11113 "0,1,2,3"  ~/anaconda3/envs/py36/bin/python " --dataset ILSVRC2012 --data_dir /home/datasets/imagenet/ILSVRC2012_dataset --model mobilenet_v3 --distributed --if-timm-dataset -b 256 --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .064 --lr-noise 0.42 0.9"

# crashed need to rerun
sh run_classification.sh 3 1 0 127.0.0.1 11113 "0,2,3"  ~/anaconda3/envs/py36/bin/python " --dataset ILSVRC2012 --data_dir /home/datasets/imagenet/ILSVRC2012_dataset --model mobilenet_v3 --distributed --if-timm-dataset -b 256 --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .05 --lr-noise 0.35 0.9"

sh run_classification.sh 3 1 0 127.0.0.1 11113 "0,1,2"  ~/anaconda3/envs/py36/bin/python " --dataset ILSVRC2012 --data_dir /home/datasets/imagenet/ILSVRC2012_dataset --model mobilenet_v3 --distributed --if-timm-dataset -b 512 --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .064  --lr-noise 0.42 0.9"


```
# kill all processes
kill $(ps aux | grep "ddp_classification.py" | grep -v grep | awk '{print $2}')
```