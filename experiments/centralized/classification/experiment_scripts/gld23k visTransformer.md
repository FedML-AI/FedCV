# ILSVRC2012-100  MobileNetV3-Large-100

# t716
PYTHON=~/miniconda3/bin/python
imagenet_data_dir=/nfs_home/datasets/ILSVRC2012
gld_data_dir=/nfs_home/datasets/landmarks
cifar10_data_dir=/nfs_home/datasets/cifar10
mnist_data_dir=/nfs_home/datasets/mnist

# DAAI
PYTHON=~/py36/bin/python
imagenet_data_dir=/home/datasets/ILSVRC2012_dataset
gld_data_dir=/home/datasets/landmarks
cifar10_data_dir=/home/datasets/cifar10
GPU_UTIL_FILE=DAAI_gpu_util.yaml
MPI_HOST_FILE=DAAI_mpi_host_file_2



## Pure with SGD
srun --cpus-per-task 4 -w hkbugpusrv04 ./single_run_classification.sh "1" ~/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/datasets/landmarks --data_transform NormalTransform --model visTransformer --pretrained --pretrained_dir ./../../../model/classification/pretrained/ViT-B_16.npz --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.001"

srun --cpus-per-task 4  ./single_run_classification.sh "1" ~/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/datasets/landmarks --data_transform NormalTransform --model visTransformer --pretrained --pretrained_dir ./../../../model/classification/pretrained/ViT-B_16.npz --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.003"

srun --cpus-per-task 4 ./single_run_classification.sh "2" ~/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/datasets/landmarks --data_transform NormalTransform --model visTransformer --pretrained --pretrained_dir ./../../../model/classification/pretrained/ViT-B_16.npz --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.01"

srun --cpus-per-task 4 ./single_run_classification.sh "3" ~/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/datasets/landmarks --data_transform NormalTransform --model visTransformer --pretrained --pretrained_dir ./../../../model/classification/pretrained/ViT-B_16.npz --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.03"




./single_run_classification.sh "0" ~/anaconda3/envs/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/comp/20481896/datasets/landmarks --data_transform NormalTransform --model visTransformer --pretrained --pretrained_dir ./../../../model/classification/pretrained/ViT-B_16.npz --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.03"

./single_run_classification.sh "0" /home/esetstore/pytorch1.4/bin/python " --dataset gld23k --partition_method homo --data_dir /home/esetstore/dataset/gld --data_transform NormalTransform --model visTransformer --pretrained --pretrained_dir ./../../../model/classification/pretrained/ViT-B_16.npz --if-timm-dataset -b 32 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.03"

./single_run_classification.sh "3" ~/anaconda3/envs/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/comp/20481896/datasets/landmarks --data_transform NormalTransform --model visTransformer --pretrained --pretrained_dir ./../../../model/classification/pretrained/ViT-B_16.npz --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --lr 0.003"

./single_run_classification.sh "0" ~/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/datasets/landmarks --data_transform NormalTransform --model visTransformer --pretrained --pretrained_dir ./../../../model/classification/pretrained/ViT-B_16.npz --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --lr 0.003"

./single_run_classification.sh "1" ~/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/datasets/landmarks --data_transform NormalTransform --model visTransformer --pretrained --pretrained_dir ./../../../model/classification/pretrained/ViT-B_16.npz --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --lr 0.01"

./single_run_classification.sh "2" ~/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/datasets/landmarks --data_transform NormalTransform --model visTransformer --pretrained --pretrained_dir ./../../../model/classification/pretrained/ViT-B_16.npz --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5  --lr 0.03"

./single_run_classification.sh "3" ~/py36/bin/python " --dataset gld23k --partition_method homo --data_dir /home/datasets/landmarks --data_transform NormalTransform --model visTransformer --pretrained --pretrained_dir ./../../../model/classification/pretrained/ViT-B_16.npz --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5  --lr 0.06"















