~/anaconda3/envs/py36/bin/python ./model_scan.py \
    --dataset gld23k \
    --model efficientnet \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \

~/anaconda3/envs/py36/bin/python ./model_scan.py \
    --dataset gld23k \
    --model mobilenet_v3 \
    --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 \

~/anaconda3/envs/py36/bin/python ./model_scan.py \
    --dataset gld23k \
    --model visTransformer \


