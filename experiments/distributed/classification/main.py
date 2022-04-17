import argparse
import logging
import os
import random
import socket
import sys
import traceback
import yaml

import numpy as np
import psutil
import setproctitle
import torch
import wandb
from mpi4py import MPI

from timm import create_model as timm_create_model
from timm.models import resume_checkpoint, load_checkpoint, convert_splitbn_model


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from FedML.fedml_api.distributed.fedavg.FedAvgAPI import FedML_init, FedML_FedAvg_distributed


from data_preprocessing.ImageNet.data_loader import load_partition_data_ImageNet
from data_preprocessing.Landmarks.data_loader import load_partition_data_landmarks
from data_preprocessing.cifar10.iid_data_loader import load_iid_cifar10
from data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from data_preprocessing.cinic10.data_loader import load_partition_data_cinic10

from training.fedavg_classification_trainer import ClassificationTrainer

from utils.context import (
    raise_MPI_error
)
from utils.logger import (
    logging_config
)


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='mobilenet', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--client_num_in_total', type=int, default=1000, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=4, metavar='NN',
                        help='number of workers')

    # parser.add_argument('--batch_size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    # parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
    #                     help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')

    parser.add_argument('--gpu_num_per_server', type=int, default=4,
                        help='gpu_num_per_server')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--gpu_util_file', type=str, default=None,
                        help='the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.')
    parser.add_argument('--gpu_util_key', type=str, default=None,
                        help='the key in gpu utilization file')
    parser.add_argument('--gpu_util_parse', type=str, default=None,
                        help='the gpu utilization string for servers and clients. If there is no \
                        gpu_util_parse, gpu will not be used. Note if this and gpu_util_file are \
                        both defined, gpu_util_parse will be used but not gpu_util_file')

    parser.add_argument('--pretrained',action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')

    parser.add_argument('--distributed', action='store_true', default=False,
                        help='If distributed training')

    parser.add_argument('--if-timm-dataset', action='store_true', default=False,
                        help='If use timm dataset augmentation')

    parser.add_argument('--data_load_num_workers', type=int, default=4,
                        help='number of workers when loading data')


    # logging settings
    parser.add_argument('--level', type=str, default='INFO',
                        help='level of logging')

    # Dataset
    parser.add_argument('--img-size', type=int, default=None, metavar='N',
                        help='Image patch size (default: None => model default)')
    parser.add_argument('--crop-pct', default=None, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--data_transform', default=None, type=str, metavar='TRANSFORM',
                        help='How to do data transform')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                        help='ratio of validation batch size to training batch size (default: 1)')


    # Model parameters
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')

    # Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')


    # Learning rate schedule parameters
    parser.add_argument('--sched', default=None, type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    # parser.add_argument('--epochs', type=int, default=200, metavar='N',
    #                     help='number of epochs to train (default: 2)')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    parser.add_argument('--decay-rounds', type=float, default=30, metavar='N',
                        help='round interval to decay LR')


    # Augmentation & regularization parameters
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    parser.add_argument('--aug-splits', type=int, default=0,
                        help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    parser.add_argument('--jsd', action='store_true', default=False,
                        help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='const',
                        help='Random erase mode (default: "const")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    parser.add_argument('--bn-tf', type=bool, default=False,
                        help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--sync-bn', action='store_true',
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist-bn', type=str, default='',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--split-bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')

    # Model Exponential Moving Average
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')


    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
    if dataset_name in ["ILSVRC2012", "ILSVRC2012-100"]:
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_ImageNet(dataset=dataset_name, data_dir=args.data_dir,
                                                 partition_method=None, partition_alpha=None,
                                                 client_number=args.client_num_in_total, 
                                                 batch_size=args.batch_size, args=args)

    elif dataset_name == "gld23k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 233
        # fed_train_map_file = os.path.join(args.data_dir, 'data_user_dict/gld23k_user_dict_train.csv')
        # fed_test_map_file = os.path.join(args.data_dir, 'data_user_dict/gld23k_user_dict_test.csv')
        fed_train_map_file = os.path.join(args.data_dir, 'mini_gld_train_split.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'mini_gld_test.csv')

        args.data_dir = os.path.join(args.data_dir, 'images')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
                                                  fed_train_map_file=fed_train_map_file,
                                                  fed_test_map_file=fed_test_map_file,
                                                  partition_method=None, partition_alpha=None,
                                                  client_number=args.client_num_in_total, 
                                                  batch_size=args.batch_size, args=args)
    elif dataset_name == "gld160k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 1262
        fed_train_map_file = os.path.join(args.data_dir, 'data_user_dict/gld160k_user_dict_train.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'data_user_dict/gld160k_user_dict_test.csv')
        args.data_dir = os.path.join(args.data_dir, 'images')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
                                                  fed_train_map_file=fed_train_map_file,
                                                  fed_test_map_file=fed_test_map_file,
                                                  partition_method=None, partition_alpha=None,
                                                  client_number=args.client_num_in_total, 
                                                  batch_size=args.batch_size, args=args)
    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            raise Exception("no such dataset")

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    if model_name == 'mobilenet_v3':
        '''model_mode \in {LARGE: 5.15M, SMALL: 2.94M}'''
        # model = MobileNetV3(model_mode='LARGE')
        model = timm_create_model(
        model_name="mobilenetv3_large_100",
        pretrained=args.pretrained,
        num_classes=output_dim,
        drop_rate=args.drop,
        # drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps)

    elif model_name == 'efficientnet':
        model = timm_create_model(
        model_name="efficientnet_b0",
        pretrained=args.pretrained,
        num_classes=output_dim,
        drop_rate=args.drop,
        # drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps)
    else:
        raise Exception("no such model")
    return model



def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    if process_ID == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = client_index % gpu_num_per_machine
        process_gpu_dict[client_index] = gpu_index

    logging.info(process_gpu_dict)
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device

def init_training_device_from_gpu_util_file(process_id, worker_number, gpu_util_file, gpu_util_key):

    if gpu_util_file == None:
        device = torch.device("cpu")
        logging.info(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.info(" ##################  Not Indicate gpu_util_file, using cpu  #################")
        logging.info(device)
        #return gpu_util_map[process_id][1]
        return device
    else:
        with open(gpu_util_file, 'r') as f:
            gpu_util_yaml = yaml.load(f, Loader=yaml.FullLoader)
            # gpu_util_num_process = 'gpu_util_' + str(worker_number)
            # gpu_util = gpu_util_yaml[gpu_util_num_process]
            gpu_util = gpu_util_yaml[gpu_util_key]
            gpu_util_map = {}
            i = 0
            for host, gpus_util_map_host in gpu_util.items():
                for gpu_j, num_process_on_gpu in enumerate(gpus_util_map_host):
                    for _ in range(num_process_on_gpu):
                        gpu_util_map[i] = (host, gpu_j)
                        i += 1
            logging.info("Process %d running on host: %s,gethostname: %s, gpu: %d ..." % (
                process_id, gpu_util_map[process_id][0], socket.gethostname(), gpu_util_map[process_id][1]))
            assert i == worker_number

        device = torch.device("cuda:" + str(gpu_util_map[process_id][1]) if torch.cuda.is_available() else "cpu")
        logging.info(device)
        #return gpu_util_map[process_id][1]
        return device

def init_training_device_from_gpu_util_parse(process_id, worker_number, gpu_util_parse):
    if gpu_util_parse == None:
        device = torch.device("cpu")
        logging.info(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.info(" ##################  Not Indicate gpu_util_file, using cpu  #################")
        logging.info(device)
        #return gpu_util_map[process_id][1]
        return device
    else:
        # example parse str `gpu_util_parse`: 
        # "gpu1:0,1,1,2;gpu2:3,3,3;gpu3:0,0,0,1,2,4,4,0"
        gpu_util_parse_temp = gpu_util_parse.split(';')
        gpu_util_parse_temp = [(item.split(':')[0], item.split(':')[1]) for item in gpu_util_parse_temp ]

        gpu_util = {}
        for (host, gpus_str) in gpu_util_parse_temp:
            gpu_util[host] = [int(num_process_on_gpu) for num_process_on_gpu in gpus_str.split(',')]

        gpu_util_map = {}
        i = 0
        for host, gpus_util_map_host in gpu_util.items():
            for gpu_j, num_process_on_gpu in enumerate(gpus_util_map_host):
                for _ in range(num_process_on_gpu):
                    gpu_util_map[i] = (host, gpu_j)
                    i += 1
        logging.info("Process %d running on host: %s,gethostname: %s, gpu: %d ..." % (
            process_id, gpu_util_map[process_id][0], socket.gethostname(), gpu_util_map[process_id][1]))
        assert i == worker_number

        device = torch.device("cuda:" + str(gpu_util_map[process_id][1]) if torch.cuda.is_available() else "cpu")
        logging.info(device)
        #return gpu_util_map[process_id][1]
        return device



if __name__ == "__main__":
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    with raise_MPI_error():
        # parse python script input parameters
        parser = argparse.ArgumentParser()
        args = add_args(parser)
        args.rank = process_id
        args.wd = args.weight_decay

        logging.info(args)

        # customize the process name
        str_process_name = 'fedavg' + " :" + str(process_id)
        setproctitle.setproctitle(str_process_name)

        logging_config(args, process_id)

        # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
        name_model_ema = "-model_ema" if args.model_ema else "-no_model_ema"
        name_aa = args.aa if args.aa is not None else "_None"
        if process_id == 0:
            wandb.init(
                entity="automl",
                project="fedcv-classification",
                name="fedavg (d)" + str(args.partition_method) + "-" +str(args.dataset)+
                    "-e" + str(args.epochs) + "-" + str(args.model) + "-" +
                    args.data_transform + "-aa" + name_aa + "-" + str(args.opt) + 
                    name_model_ema + "-bs" + str(args.batch_size) +
                    "-lr" + str(args.lr) + "-wd" + str(args.wd),
                config=args
            )

        # Set the random seed. The np.random seed determines the dataset partition.
        # The torch_manual_seed determines the initial weight.
        # We fix these two, so that we can reproduce the result.
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        # GPU arrangement: Please customize this function according your own topology.
        # The GPU server list is configured at "mpi_host_file".
        # If we have 4 machines and each has two GPUs, and your FL network has 8 workers and a central worker.
        # The 4 machines will be assigned as follows:
        # machine 1: worker0, worker4, worker8;
        # machine 2: worker1, worker5;
        # machine 3: worker2, worker6;
        # machine 4: worker3, worker7;
        # Therefore, we can see that workers are assigned according to the order of machine list.
        logging.info("process_id = %d, size = %d" % (process_id, worker_number))
        if args.gpu_util_parse is not None:
            device = init_training_device_from_gpu_util_parse(process_id, worker_number, args.gpu_util_parse)
        else:
            device = init_training_device_from_gpu_util_file(process_id, worker_number, args.gpu_util_file, args.gpu_util_key)

        # load data
        dataset = load_data(args, args.dataset)
        [train_data_num, test_data_num, train_data_global, test_data_global,
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

        # create model.
        # Note if the model is DNN (e.g., ResNet), the training will be very slow.
        # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
        model = create_model(args, model_name=args.model, output_dim=dataset[7])

        model_trainer = ClassificationTrainer(model, device, args)
        FedML_FedAvg_distributed(process_id, worker_number, device, comm,
                                model, train_data_num, train_data_global, test_data_global,
                                train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args, model_trainer)






