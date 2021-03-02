
## SGD
### 10 clients
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
