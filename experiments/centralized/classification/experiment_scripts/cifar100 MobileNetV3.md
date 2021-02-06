# CIFAR100  MobileNetV3-Large-100


# DAAI
PYTHON=~/py36/bin/python
imagenet_data_dir=/home/datasets/ILSVRC2012_dataset
gld_data_dir=/home/datasets/landmarks
cifar10_data_dir=/home/datasets/cifar10
cifar100_data_dir=/home/datasets/cifar100
GPU_UTIL_FILE=DAAI_gpu_util.yaml
MPI_HOST_FILE=DAAI_mpi_host_file_2



# Pure
./single_run_classification.sh "1"  ~/py36/bin/python " --dataset cifar100 --data_dir /home/datasets/cifar100 --data_transform FLTransform --model mobilenet_v3 --if-timm-dataset -b 256 --sched step --epochs 100 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --remode pixel --reprob 0.2 --lr 0.03"






