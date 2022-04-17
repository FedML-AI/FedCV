import os
import sys
import time
import logging
import collections
import csv

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from timm.data import Dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset

from .datasets import Landmarks

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _read_csv(path: str):
  """Reads a csv file, and returns the content inside a list of dictionaries.
  Args:
    path: The path to the csv file.
  Returns:
    A list of dictionaries. Each row in the csv file will be a list entry. The
    dictionary is keyed by the column names.
  """
  with open(path, 'r') as f:
    return list(csv.DictReader(f))



class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_landmarks(args):

    if args.data_transform == 'FLTransform':
        IMAGENET_MEAN = [0.5, 0.5, 0.5]
        IMAGENET_STD = [0.5, 0.5, 0.5]
    elif args.data_transform == 'NormalTransform':
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
    else:
        raise NotImplementedError

    image_size = 224
    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, valid_transform



def get_mapping_per_user(fn):
    """
    mapping_per_user is {'user_id': [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ... {}], 
                         'user_id': [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ... {}],
    } or               
                        [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ...  
                         {'user_id': xxx, 'image_id': xxx, 'class': xxx} ... ]
    }
    """
    mapping_table = _read_csv(fn)
    expected_cols = ['user_id', 'image_id', 'class']
    if not all(col in mapping_table[0].keys() for col in expected_cols):
        logger.error('%s has wrong format.', fn)
        raise ValueError(
            'The mapping file must contain user_id, image_id and class columns. '
            'The existing columns are %s' % ','.join(mapping_table[0].keys()))

    data_local_num_dict = dict()


    mapping_per_user = collections.defaultdict(list)
    data_files = []
    net_dataidx_map = {}
    sum_temp = 0

    for row in mapping_table:
        user_id = row['user_id']
        mapping_per_user[user_id].append(row)
    for user_id, data in mapping_per_user.items():
        num_local = len(mapping_per_user[user_id])
        # net_dataidx_map[user_id]= (sum_temp, sum_temp+num_local)
        # data_local_num_dict[user_id] = num_local
        net_dataidx_map[int(user_id)]= (sum_temp, sum_temp+num_local)
        data_local_num_dict[int(user_id)] = num_local
        sum_temp += num_local
        data_files += mapping_per_user[user_id]
    assert sum_temp == len(data_files)

    return data_files, data_local_num_dict, net_dataidx_map


def get_dataloader(dataset_train, dataset_test, dataidxs=None, args=None):
    train_bs = args.batch_size
    test_bs = args.batch_size

    train_dl = data.DataLoader(dataset=dataset_train, batch_size=train_bs, shuffle=True, drop_last=False,
                        pin_memory=True, num_workers=args.data_load_num_workers)
    test_dl = data.DataLoader(dataset=dataset_test, batch_size=test_bs, shuffle=False, drop_last=False,
                        pin_memory=True, num_workers=args.data_load_num_workers)

    return train_dl, test_dl


# def get_dataloader_Landmarks(datadir, train_files, test_files, train_bs, test_bs, dataidxs=None):
#     dl_obj = Landmarks

#     transform_train, transform_test = _data_transforms_landmarks()

#     train_ds = dl_obj(datadir, train_files, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
#     test_ds = dl_obj(datadir, test_files, dataidxs=dataidxs, train=False, transform=transform_test, download=True)

#     train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
#     test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

#     return train_dl, test_dl


def get_timm_loader(dataset_train, dataset_test, args):
    """
        Use for get data loader of timm, for data transforms, augmentations, etc.
        dataset: self-defined dataset,
        return: timm loader
    """
    logging.info("Using timm dataset and dataloader")

    # TODO not sure whether any problem here
    data_config = resolve_data_config(vars(args), model=None, verbose=args.rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']

    # some args not in the args
    args.prefetcher = False
    args.pin_mem = False
    collate_fn = None
    args.use_multi_epochs_loader = False

    train_batch_size = args.batch_size
    test_batch_size = args.batch_size // 4

    if args.data_transform == 'FLTransform':
        data_config['mean'] = [0.5, 0.5, 0.5]
        data_config['std'] = [0.5, 0.5, 0.5]
    elif args.data_transform == 'NormalTransform':
        pass 
        # data_config['mean'] = 
        # data_config['std'] = 
    else:
        raise NotImplementedError

    logging.info("data transform, MEAN: {}, STD: {}.".format(
        data_config['mean'], data_config['std']))
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=train_batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.data_load_num_workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    loader_eval = create_loader(
        dataset_test,
        input_size=data_config['input_size'],
        batch_size=test_batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.data_load_num_workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )
    return loader_train, loader_eval



def load_partition_data_landmarks(dataset, data_dir, fed_train_map_file, fed_test_map_file, 
                            partition_method=None, partition_alpha=None, client_number=233, batch_size=10, args=None):

    train_files, data_local_num_dict, net_dataidx_map = get_mapping_per_user(fed_train_map_file)
    test_files = _read_csv(fed_test_map_file)

    class_num = len(np.unique([item['class'] for item in train_files]))
    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = len(train_files)


    transform_train, transform_test = _data_transforms_landmarks(args)

    train_dataset = Landmarks(data_dir, train_files, dataidxs=None, train=True, transform=transform_train, download=True)
    test_dataset = Landmarks(data_dir, test_files, dataidxs=None, train=False, transform=transform_test, download=True)


    if args.if_timm_dataset:
        train_data_global, test_data_global = get_timm_loader(train_dataset, test_dataset, args)
    else:
        train_data_global, test_data_global = get_dataloader(train_dataset, test_dataset, args)

    # logging.info("train_dl_global number = " + str(len(train_data_global)))
    # logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_files)

    # get local dataset
    data_local_num_dict = data_local_num_dict
    train_data_local_dict = dict()
    test_data_local_dict = dict()


    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        # local_data_num = len(dataidxs)
        local_data_num = dataidxs[1] - dataidxs[0]
        # data_local_num_dict[client_idx] = local_data_num
        # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        train_dataset_local = Landmarks(data_dir, train_files, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_dataset_local = Landmarks(data_dir, test_files, dataidxs=None, train=False, transform=transform_test, download=True)
        if args.if_timm_dataset:
            train_data_local, test_data_local = get_timm_loader(train_dataset_local, test_dataset_local, args)
        else:
            train_data_local, test_data_local = get_dataloader(train_dataset_local, test_dataset_local, args)

        # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #     client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    # logging("data_local_num_dict: %s" % data_local_num_dict)
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


if __name__ == '__main__':
    data_dir = './cache/images'
    fed_g23k_train_map_file = '../../../data/gld/data_user_dict/gld23k_user_dict_train.csv'
    fed_g23k_test_map_file = '../../../data/gld/data_user_dict/gld23k_user_dict_test.csv'

    fed_g160k_train_map_file = '../../../data/gld/data_user_dict/gld160k_user_dict_train.csv'
    fed_g160k_map_file = '../../../data/gld/data_user_dict/gld160k_user_dict_test.csv'

    dataset_name = 'g160k'

    if dataset_name == 'g23k':
        client_number = 233
        fed_train_map_file = fed_g23k_train_map_file
        fed_test_map_file = fed_g23k_test_map_file
    elif dataset_name == 'g160k':
        client_number = 1262 
        fed_train_map_file = fed_g160k_train_map_file
        fed_test_map_file = fed_g160k_map_file

    train_data_num, test_data_num, train_data_global, test_data_global, \
        data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = \
        load_partition_data_landmarks(None, data_dir, fed_train_map_file, fed_test_map_file, 
                            partition_method=None, partition_alpha=None, client_number=client_number, batch_size=10)

    print(train_data_num, test_data_num, class_num)
    print(data_local_num_dict)

    i = 0
    for data, label in train_data_global:
        print(data)
        print(label)
        i += 1
        if i > 5:
            break
    print("=============================\n")

    for client_idx in range(client_number):
        i = 0
        for data, label in train_data_local_dict[client_idx]:
            print(data)
            print(label)
            i += 1
            if i > 5:
                break



