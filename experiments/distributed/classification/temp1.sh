


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

