mpirun -np 11 -host scigpu10:11 \
        ~/anaconda3/envs/py36/bin/python ./main.py \
            --gpu_util_parse "scigpu10:0,4,4,3" \
                --client_num_per_round 10 --client_num_in_total 100 \
                    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
                        --frequency_of_the_test 100 \
                            --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
                                --if-timm-dataset -b 32  --data_transform FLTransform \
                                    --comm_round 4000  --epochs 1 \
                                        --model efficientnet --pretrained\
                                            --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
                                                --opt sgd --lr 0.001 --warmup-lr 1e-6 --weight-decay 1e-5 \
                                                    --sched StepLR --decay-rounds 1 --decay-rate .999

mpirun -np 11 -host scigpu10:11 \
        ~/anaconda3/envs/py36/bin/python ./main.py \
            --gpu_util_parse "scigpu10:0,4,4,3" \
                --client_num_per_round 10 --client_num_in_total 100 \
                    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
                        --frequency_of_the_test 100 \
                            --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
                                --if-timm-dataset -b 32  --data_transform FLTransform \
                                    --comm_round 4000  --epochs 1 \
                                        --model efficientnet --pretrained \
                                            --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
                                                --opt sgd --lr 0.003 --warmup-lr 1e-6 --weight-decay 1e-5 \
                                                    --sched StepLR --decay-rounds 1 --decay-rate .999


mpirun -np 11 -host scigpu10:11 \
        ~/anaconda3/envs/py36/bin/python ./main.py \
            --gpu_util_parse "scigpu10:0,4,4,3" \
                --client_num_per_round 10 --client_num_in_total 100 \
                    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
                        --frequency_of_the_test 100 \
                            --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
                                --if-timm-dataset -b 32  --data_transform FLTransform \
                                    --comm_round 4000  --epochs 1 \
                                        --model efficientnet --pretrained\
                                            --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
                                                --opt sgd --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
                                                    --sched StepLR --decay-rounds 1 --decay-rate .999

mpirun -np 11 -host scigpu10:11 \
        ~/anaconda3/envs/py36/bin/python ./main.py \
            --gpu_util_parse "scigpu10:0,4,4,3" \
                --client_num_per_round 10 --client_num_in_total 100 \
                    --gpu_server_num 1 --gpu_num_per_server 1 --ci 0 \
                        --frequency_of_the_test 100 \
                            --dataset cifar100 --data_dir ~/datasets/cifar100 --partition_method hetero --partition_alpha 100 \
                                --if-timm-dataset -b 32  --data_transform FLTransform \
                                    --comm_round 4000  --epochs 1 \
                                        --model efficientnet --pretrained \
                                            --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
                                                --opt sgd --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
                                                    --sched StepLR --decay-rounds 1 --decay-rate .999





