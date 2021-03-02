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
./single_run_classification.sh "0" ~/py36/bin/python " --dataset gld23k --data_dir /home/datasets/landmarks --data_transform FLTransform --model visTransformer --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --lr 0.003"

./single_run_classification.sh "1" ~/py36/bin/python " --dataset gld23k --data_dir /home/datasets/landmarks --data_transform FLTransform --model visTransformer --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5 --lr 0.01"

./single_run_classification.sh "2" ~/py36/bin/python " --dataset gld23k --data_dir /home/datasets/landmarks --data_transform FLTransform --model visTransformer --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5  --lr 0.03"

./single_run_classification.sh "3" ~/py36/bin/python " --dataset gld23k --data_dir /home/datasets/landmarks --data_transform FLTransform --model visTransformer --if-timm-dataset -b 256 --sched step --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt momentum --warmup-lr 1e-6 --weight-decay 1e-5  --lr 0.06"















