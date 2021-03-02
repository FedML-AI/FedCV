# ILSVRC2012-100  MobileNetV3-Large-100

# scigpu
PYTHON=~/anaconda3/envs/py36/bin/python
imagenet_data_dir=/home/datasets/imagenet/ILSVRC2012_dataset
gld_data_dir=~/datasets/landmarks
cifar10_data_dir=~/datasets/cifar10
GPU_UTIL_FILE=scigpu_gpu_util.yaml
MPI_HOST_FILE=scigpu_mpi_host_file

# DAAI
PYTHON=~/py36/bin/python
imagenet_data_dir=/home/datasets/ILSVRC2012_dataset
gld_data_dir=/home/datasets/landmarks
cifar10_data_dir=/home/datasets/cifar10
GPU_UTIL_FILE=DAAI_gpu_util.yaml
MPI_HOST_FILE=DAAI_mpi_host_file_2





## Pure with SGD
```
# running
./single_run_classification.sh "0" ~/py36/bin/python " --dataset ILSVRC2012-100 --data_dir /home/datasets/ILSVRC2012_dataset --data_transform FLTransform --model mobilenet_v3 --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 -lr 0.003"

./single_run_classification.sh "1" ~/py36/bin/python " --dataset ILSVRC2012-100 --data_dir /home/datasets/ILSVRC2012_dataset --data_transform FLTransform --model mobilenet_v3 --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5  --lr 0.01"

./single_run_classification.sh "2" ~/py36/bin/python " --dataset ILSVRC2012-100 --data_dir /home/datasets/ILSVRC2012_dataset --data_transform FLTransform --model mobilenet_v3 --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5   --lr 0.03"

./single_run_classification.sh "3" ~/py36/bin/python " --dataset ILSVRC2012-100 --data_dir /home/datasets/ILSVRC2012_dataset --data_transform FLTransform --model mobilenet_v3 --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5   --lr 0.06"


