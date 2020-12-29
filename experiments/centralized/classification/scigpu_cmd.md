
# PyTorch DDP classification

## lr_scheduler parameter reference:


EfficientNet-B0 with RandAugment - 77.7 top-1, 95.3 top-5



```

sh run_classification.sh 2 1 0 localhost 11111 "0,3"  ~/anaconda3/envs/py36/bin/python " --dataset ILSVRC2012 --data_dir /home/datasets/imagenet/ILSVRC2012_dataset --epochs 450  --model efficientnet -b 384 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.2  --lr 0.048"

# 384 is out of memory

sh run_classification.sh 2 1 0 localhost 11111 "0,3"  ~/anaconda3/envs/py36/bin/python " --dataset ILSVRC2012 --data_dir /home/datasets/imagenet/ILSVRC2012_dataset --epochs 450  --model efficientnet -b 256 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.2  --lr 0.048"
```













```
# kill all processes
kill $(ps aux | grep "ddp_classification.py" | grep -v grep | awk '{print $2}')
```