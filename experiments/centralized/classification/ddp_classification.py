import argparse
import logging
import os
import random
import socket
import sys
import traceback


import numpy as np
import psutil
import setproctitle
import wandb
from mpi4py import MPI
from timm import create_model as timm_create_model
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from utils.tracker import RuntimeTracker
from utils.metrics import Metrics
from utils.wandb_util import wandb_log
from data_preprocessing.ImageNet.data_loader import distributed_centralized_ImageNet_loader
from data_preprocessing.Landmarks.data_loader import load_partition_data_landmarks
from training.classification_trainer import ClassificationTrainer



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

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

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

    parser.add_argument('--local_rank', type=int, default=0,
                        help='given by torch.distributed.launch')

    parser.add_argument('--pretrained', type=bool, default=False,
                        help='Start with pretrained version of specified network (if avail)')


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
    parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
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
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 2)')
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

    # Augmentation & regularization parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    parser.add_argument('--drop_rate', type=float, default=0.0,
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_connect_rate', type=float, default=None,
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop_path_rate', type=float, default=None,
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop_block_rate', type=float, default=None,
                        help='Drop block rate (default: None)')
    parser.add_argument('--global_pool', type=float, default=0,
                        help='CI')

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



    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
    if dataset_name == "ILSVRC2012":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = distributed_centralized_ImageNet_loader(dataset=dataset_name, data_dir=args.data_dir,
                                                 partition_method=None,
                                                 client_number=args.client_num_in_total, batch_size=args.batch_size)

    elif dataset_name == "gld23k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 233
        fed_train_map_file = os.path.join(args.data_dir, 'data_user_dict/gld23k_user_dict_train.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'data_user_dict/gld23k_user_dict_test.csv')
        args.data_dir = os.path.join(args.data_dir, 'images')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
                                                  fed_train_map_file=fed_train_map_file,
                                                  fed_test_map_file=fed_test_map_file,
                                                  partition_method=None, partition_alpha=None,
                                                  client_number=args.client_num_in_total, batch_size=args.batch_size)

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
                                                  client_number=args.client_num_in_total, batch_size=args.batch_size)
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
        drop_rate=args.drop_rate,
        drop_connect_rate=args.drop_connect_rate,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path_rate,
        drop_block_rate=args.drop_block_rate,
        global_pool=args.global_pool,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps)

    elif model_name == 'efficientnet':
        model = timm_create_model(
        model_name="efficientnet_b0",
        pretrained=args.pretrained,
        num_classes=output_dim,
        drop_rate=args.drop_rate,
        drop_connect_rate=args.drop_connect_rate,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path_rate,
        drop_block_rate=args.drop_block_rate,
        global_pool=args.global_pool,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps)
    else:
        raise Exception("no such model")
    return model


def init_ddp():
    # use InfiniBand
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_IFNAME'] = 'ib0'

    # This the global rank: 0, 1, 2, ..., 15
    global_rank = int(os.environ['RANK'])
    print("int(os.environ['RANK']) = %d" % global_rank)

    # This the globak world_size
    world_size = int(os.environ['WORLD_SIZE'])
    print("world_size = %d" % world_size)

    # initialize the process group
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    local_rank = args.local_rank
    print(f"Running basic DDP example on local rank {local_rank}.")
    return local_rank, global_rank


def get_ddp_model(model, local_rank):
    return DDP(model, device_ids=[local_rank], output_device=local_rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DDP Demo")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    logging.info(args)
    args.weight_decay = args.wd


    # DDP
    local_rank, global_rank = init_ddp()
    process_id = global_rank

    # customize the process name
    str_process_name = "ddp_classification:" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    while logging.getLogger().handlers:
        logging.getLogger().handlers.clear()
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(str(process_id) + 
        ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    console.setFormatter(formatter)
    # Create an instance
    logging.getLogger().addHandler(console)
    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                ", host name = " + hostname + "########" +
                ", process ID = " + str(os.getpid()) +
                ", process Name = " + str(psutil.Process(os.getpid())))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            entity="automl",
            project="fedcv-classification",
            name="FedCV (c new)" + str(args.partition_method) + "-" +str(args.dataset)+
                "-e" + str(args.epochs) + "-" + str(args.model) + "-" +
                str(args.client_optimizer) + "-bs" + str(args.batch_size) +
                "-lr" + str(args.lr) + "-wd" + str(args.wd),
            config=args
        )

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # GPU
    device = torch.device("cuda:" + str(local_rank))

    # load data
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    model = get_ddp_model(model, local_rank)
    if global_rank == 0:
        print(model)

    metrics = Metrics(topks=[1], task="classification")
    train_tracker = RuntimeTracker(things_to_track=metrics.metric_names)
    test_tracker = RuntimeTracker(things_to_track=metrics.metric_names)

    model_trainer = ClassificationTrainer(model)
    for epoch in range(args.epochs):
        model_trainer.train_one_epoch(train_data_global)
        model_trainer.test(test_data_global, metrics, test_tracker)
        wandb_log(prefix='Test', sp_values=test_tracker(), com_values={"epoch": epoch})

    dist.destroy_process_group()
