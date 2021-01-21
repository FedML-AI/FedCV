# ILSVRC2012-100










# 10 clients
```

# DAAI
# srun -N2 -B 4-4:2-2 \
# srun -w hkbugpusrv03 -n 21 -B 21:4 \
salloc -w hkbugpusrv03 -n 21 --cpus-per-task=4 \
mpiexec \
    ~/py36/bin/python ./main.py \
    --gpu_util_parse "hkbugpusrv03:6,5,5,5" \
    --client_num_per_round 20 --client_num_in_total 100 \
    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
    --frequency_of_the_test 10 \
    --dataset ILSVRC2012-100 --data_dir /home/datasets/ILSVRC2012_dataset \
    --if-timm-dataset -b 16  --data_transform FLTransform \
    --comm_round 1000  --epochs 1 \
    --model mobilenet_v3 \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
    --opt rmsproptf --lr 0.001 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
    --sched step --decay-rounds 1 --decay-rate .992

```







