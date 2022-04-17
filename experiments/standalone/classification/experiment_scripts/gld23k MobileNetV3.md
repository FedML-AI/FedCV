# gld23k



# 10 clients
```

# DAAI

~/py36/bin/python ./main.py \
--gpu 0 \
--client_num_per_round 4 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 1000  --epochs 1 \
--model mobilenet_v3 \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.01 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .992

~/py36/bin/python ./main.py \
--gpu 3 \
--client_num_per_round 4 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 1000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.01 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .992

# =========================================================================
# high decay-rate

~/py36/bin/python ./main.py \
--gpu 2 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 128  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model mobilenet_v3 \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.03 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .997

~/py36/bin/python ./main.py \
--gpu 3 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 128  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model mobilenet_v3 \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.03 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .997

~/py36/bin/python ./main.py \
--gpu 0 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model mobilenet_v3 \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.03 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .999

~/py36/bin/python ./main.py \
--gpu 1 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model mobilenet_v3 \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.01 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .999

~/py36/bin/python ./main.py \
--gpu 2 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 128  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model mobilenet_v3 \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.03 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .999

~/py36/bin/python ./main.py \
--gpu 3 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 128  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model mobilenet_v3 \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.1 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .999


~/py36/bin/python ./main.py \
--gpu 0 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 3000  --epochs 1 \
--model mobilenet_v3 \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.01 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .9992

~/py36/bin/python ./main.py \
--gpu 1 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 128  --data_transform FLTransform \
--comm_round 3000  --epochs 1 \
--model mobilenet_v3 \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.03 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .9992

~/py36/bin/python ./main.py \
--gpu 2 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 16  --data_transform FLTransform \
--comm_round 3000  --epochs 1 \
--model mobilenet_v3 \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.01 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .9992

~/py36/bin/python ./main.py \
--gpu 3 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 16  --data_transform FLTransform \
--comm_round 3000  --epochs 1 \
--model mobilenet_v3 \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.005 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .9992


```














