
# PyTorch DDP classification

## lr_scheduler parameter reference:


## Imagenet-1000
EfficientNet-B0 with RandAugment - 77.7 top-1, 95.3 top-5



```

sh run_classification.sh 8 1 0 127.0.0.1 11111 "0,3"  ~/py36/bin/python " --dataset ILSVRC2012 --data_dir /home/datasets/ILSVRC2012_dataset --model efficientnet --distributed --if-timm-dataset -b 256 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .048"
```


MobileNetV3-Large-100 - 75.766 top-1, 92,542 top-5



```
# crashed need to rerun
sh run_classification.sh 4 1 0 127.0.0.1 11113 "0,1,2,3" ~/py36/bin/python " --dataset ILSVRC2012 --data_dir /home/datasets/ILSVRC2012_dataset --model mobilenet_v3 --distributed --if-timm-dataset -b 256 --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .064 --lr-noise 0.42 0.9"
```

## Imagenet-100




## gld160k
EfficientNet-B0 with RandAugment


```
./single_run_classification.sh "0"  ~/py36/bin/python " --dataset gld160k --data_dir /home/datasets/landmarks --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.003"

./single_run_classification.sh "1"  ~/py36/bin/python " --dataset gld160k --data_dir /home/datasets/landmarks --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.03"


./single_run_classification.sh "2"  ~/py36/bin/python " --dataset gld160k --data_dir /home/datasets/landmarks --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.02"


./single_run_classification.sh "3"  ~/py36/bin/python " --dataset gld160k --data_dir /home/datasets/landmarks --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr 0.005"


./single_run_classification.sh "0"  ~/py36/bin/python " --dataset gld160k --data_dir /home/datasets/landmarks --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .048"

./single_run_classification.sh "1"  ~/py36/bin/python " --dataset gld160k --data_dir /home/datasets/landmarks --model efficientnet --if-timm-dataset -b 256 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .1"



```


MobileNetV3-Large-100
```
./single_run_classification.sh "2"  ~/py36/bin/python " --dataset gld160k --data_dir /home/datasets/landmarks --model mobilenet_v3 --if-timm-dataset -b 256 --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .048 --lr-noise 0.42 0.9"



./single_run_classification.sh "3"  ~/py36/bin/python " --dataset gld160k --data_dir /home/datasets/landmarks --model mobilenet_v3 --if-timm-dataset -b 256 --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .064 --lr-noise 0.42 0.9"


```






## gld23k
EfficientNet-B0 with RandAugment


```
./single_run_classification.sh "0"  ~/py36/bin/python " --dataset gld23k --data_dir /home/datasets/landmarks --model efficientnet  --if-timm-dataset -b 256 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .048"
```


MobileNetV3-Large-100
```
./single_run_classification.sh "0"  ~/py36/bin/python " --dataset gld23k --data_dir /home/datasets/landmarks --model mobilenet_v3  --if-timm-dataset -b 256 --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .064 --lr-noise 0.42 0.9"
```








# kill all processes
```
kill $(ps aux | grep "ddp_classification.py" | grep -v grep | awk '{print $2}')

```
# kill all processes
```
kill $(ps aux | grep "classification" | grep -v grep | awk '{print $2}')
```
# kill all processes
```
kill $(ps aux | grep "481896/py36/bin/python" | grep -v grep | awk '{print $2}')
```









