# gld23k


# 4 clients
```

# DAAI


~/py36/bin/python ./main.py \
--gpu 0 \
--client_num_per_round 4 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.01 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .992

~/py36/bin/python ./main.py \
--gpu 2 \
--client_num_per_round 4 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.03 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .992

~/py36/bin/python ./main.py \
--gpu 3 \
--client_num_per_round 4 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.003 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .992



~/py36/bin/python ./main.py \
--gpu 0 \
--client_num_per_round 4 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 128  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.03 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .992


~/py36/bin/python ./main.py \
--gpu 1 \
--client_num_per_round 4 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 128  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.01 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .992
```


# 10 clients

```
~/py36/bin/python ./main.py \
--gpu 2 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 128  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.01 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .992

~/py36/bin/python ./main.py \
--gpu 3 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 128  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.03 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .992

~/py36/bin/python ./main.py \
--gpu 1 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 128  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.03 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .997


~/py36/bin/python ./main.py \
--gpu 2 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 128  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.01 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .997

~/py36/bin/python ./main.py \
--gpu 3 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model mobilenet_v3 \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.01 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .997


# =========================================================================
# high decay-rate

~/py36/bin/python ./main.py \
--gpu 0 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.03 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .999

~/py36/bin/python ./main.py \
--gpu 1 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 128  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.03 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .999

~/py36/bin/python ./main.py \
--gpu 2 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.01 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .999

~/py36/bin/python ./main.py \
--gpu 3 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 128  --data_transform FLTransform \
--comm_round 2000  --epochs 1 \
--model efficientnet \
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
--model efficientnet \
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
--model efficientnet \
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
--model efficientnet \
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
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt rmsproptf --lr 0.005 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .9992

```




## Add tricks
### SGD
~/py36/bin/python ./main.py \
--gpu 2 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 4000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt momentum --lr 0.003 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .999 \
--model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 





~/py36/bin/python ./main.py \
--gpu 2 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 4000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt momentum --lr 0.01 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .999 \
--model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 


~/py36/bin/python ./main.py \
--gpu 2 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 4000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt momentum --lr 0.03 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .999 \
--model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 



~/py36/bin/python ./main.py \
--gpu 3 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 4000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt momentum --lr 0.3 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .999 \
--model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 



~/py36/bin/python ./main.py \
--gpu 3 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 4000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt momentum --lr 0.1 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .999 \
--model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 


~/py36/bin/python ./main.py \
--gpu 3 \
--client_num_per_round 10 --client_num_in_total 233 \
--frequency_of_the_test 10 \
--dataset gld23k --data_dir /home/datasets/landmarks \
--if-timm-dataset -b 32  --data_transform FLTransform \
--comm_round 4000  --epochs 1 \
--model efficientnet \
--drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \
--opt momentum --lr 0.6 --warmup-lr 1e-6 --weight-decay 1e-5 \
--sched step --decay-rounds 1 --decay-rate .999 \
--model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 

















