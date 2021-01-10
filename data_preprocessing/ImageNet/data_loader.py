import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from timm.data import Dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset

from .datasets import ImageNet
from .datasets import ImageNet100
from .datasets import ImageNet_truncated
from .datasets_hdf5 import ImageNet_hdf5
from .datasets_hdf5 import ImageNet_truncated_hdf5


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


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


def _data_transforms_ImageNet(args):
    # IMAGENET_MEAN = [0.5071, 0.4865, 0.4409]
    # IMAGENET_STD = [0.2673, 0.2564, 0.2762]
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



def get_ImageNet_truncated(imagenet_dataset_train, imagenet_dataset_test, train_bs,
                                      test_bs, dataidxs=None, net_dataidx_map=None, args=None):
    """
        imagenet_dataset_train, imagenet_dataset_test should be ImageNet or ImageNet_hdf5
    """
    if type(imagenet_dataset_train) == ImageNet:
        dl_obj = ImageNet_truncated
    elif type(imagenet_dataset_train) == ImageNet_hdf5:
        dl_obj = ImageNet_truncated_hdf5
    else:
        raise NotImplementedError()

    transform_train, transform_test = _data_transforms_ImageNet(args)

    train_ds = dl_obj(imagenet_dataset_train, dataidxs, net_dataidx_map, train=True, transform=transform_train,
                      download=False)
    test_ds = dl_obj(imagenet_dataset_test, dataidxs, net_dataidx_map, train=False, transform=transform_test,
                     download=False)
    return train_ds, test_ds


def get_dataloader(dataset_train, dataset_test, train_bs,
                    test_bs, dataidxs=None, net_dataidx_map=None):

    train_dl = data.DataLoader(dataset=dataset_train, batch_size=train_bs, shuffle=True, drop_last=False,
                        pin_memory=True, num_workers=args.data_load_num_workers)
    test_dl = data.DataLoader(dataset=dataset_test, batch_size=test_bs, shuffle=False, drop_last=False,
                        pin_memory=True, num_workers=args.data_load_num_workers)

    return train_dl, test_dl



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


def distributed_centralized_ImageNet_loader(dataset, data_dir,
                        world_size, rank, batch_size, args):
    """
        Used for generating distributed dataloader for 
        accelerating centralized training 
    """

    train_bs=batch_size
    test_bs=batch_size

    transform_train, transform_test = _data_transforms_ImageNet(args)
    if dataset == 'ILSVRC2012':
        train_dataset = ImageNet(data_dir=data_dir,
                                dataidxs=None,
                                train=True,
                                transform=transform_train) 

        test_dataset = ImageNet(data_dir=data_dir,
                                dataidxs=None,
                                train=False,
                                transform=transform_test)
        class_num = 1000
    elif dataset == 'ILSVRC2012-100':
        train_dataset = ImageNet100(data_dir=data_dir,
                                dataidxs=None,
                                train=True,
                                transform=transform_train) 

        test_dataset = ImageNet100(data_dir=data_dir,
                                dataidxs=None,
                                train=False,
                                transform=transform_test) 
        class_num = 100
    elif dataset == 'ILSVRC2012_hdf5':
        train_dataset = ImageNet_hdf5(data_dir=data_dir,
                                dataidxs=None,
                                train=True,
                                transform=transform_train) 

        test_dataset = ImageNet_hdf5(data_dir=data_dir,
                                dataidxs=None,
                                train=False,
                                transform=transform_test) 
        class_num = 1000
    else:
        raise NotImplementedError


    if args.if_timm_dataset:
        train_dl, test_dl = get_timm_loader(train_dataset, test_dataset, args)
    else:
        train_sam = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        # test_sam = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

        train_dl = data.DataLoader(train_dataset, batch_size=train_bs , sampler=train_sam,
                            pin_memory=True, num_workers=args.data_load_num_workers)

        test_dl = data.DataLoader(test_dataset, batch_size=test_bs, sampler=None,
                            pin_memory=True, num_workers=args.data_load_num_workers)

    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)

    logging.info("len of train_dataset: {}".format(train_data_num))
    logging.info("len of test_dataset: {}".format(test_data_num))

    return train_data_num, test_data_num, train_dl, test_dl, \
           None, None, None, class_num


def load_partition_data_ImageNet(dataset, data_dir, partition_method=None, partition_alpha=None, 
                                    client_number=100, batch_size=10, args=None):

    transform_train, transform_test = _data_transforms_ImageNet(args)
    if dataset == 'ILSVRC2012':
        train_dataset = ImageNet(data_dir=data_dir,
                                dataidxs=None,
                                train=True,
                                transform=transform_train) 

        test_dataset = ImageNet(data_dir=data_dir,
                                dataidxs=None,
                                train=False,
                                transform=transform_test)
        class_num = 1000
    elif dataset == 'ILSVRC2012-100':
        train_dataset = ImageNet100(data_dir=data_dir,
                                dataidxs=None,
                                train=True,
                                transform=transform_train) 

        test_dataset = ImageNet100(data_dir=data_dir,
                                dataidxs=None,
                                train=False,
                                transform=transform_test) 
        class_num = 100
    elif dataset == 'ILSVRC2012_hdf5':
        train_dataset = ImageNet_hdf5(data_dir=data_dir,
                                dataidxs=None,
                                train=True,
                                transform=transform_train) 

        test_dataset = ImageNet_hdf5(data_dir=data_dir,
                                dataidxs=None,
                                train=False,
                                transform=transform_test) 
        class_num = 1000
    else:
        raise NotImplementedError

    net_dataidx_map = train_dataset.get_net_dataidx_map()


    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    # train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)
    class_num_dict = train_dataset.get_data_local_num_dict()

    if args.if_timm_dataset:
        train_data_global, test_data_global = get_timm_loader(train_dataset, test_dataset, args)
    else:
        train_data_global, test_data_global = get_dataloader(train_dataset, test_dataset,
                                                                                train_bs=batch_size, test_bs=batch_size,
                                                                                dataidxs=None, net_dataidx_map=None)

    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        if client_number == 1000:
            if dataset not in ['ILSVRC2012', 'ILSVRC2012_hdf5']:
                raise NotImplementedError("Only support 1000 clients for Full ILSVRC2012!")
            dataidxs = client_idx
            data_local_num_dict = class_num_dict
        elif client_number == 100:
            if dataset in ['ILSVRC2012', 'ILSVRC2012_hdf5']:
                dataidxs = [client_idx * 10 + i for i in range(10)]
                data_local_num_dict[client_idx] = sum(class_num_dict[client_idx + i] for i in range(10))
            elif dataset in ['ILSVRC2012-100']:
                dataidxs = client_idx
                data_local_num_dict = class_num_dict
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError("Not support other client_number for now!")

        local_data_num = data_local_num_dict[client_idx]

        # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        # train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
        #                                          dataidxs)
        train_dataset_local, test_dataset_local = get_ImageNet_truncated(train_dataset, test_dataset,
                                                                        train_bs=batch_size, test_bs=batch_size,
                                                                        dataidxs=dataidxs,
                                                                        net_dataidx_map=net_dataidx_map, args=args)
        if args.if_timm_dataset:
            train_data_local, test_data_local = get_timm_loader(train_dataset_local, test_dataset_local, args)
        else:
            train_data_local, test_data_local = get_dataloader(train_dataset_local, test_dataset_local,
                                                                train_bs=batch_size, test_bs=batch_size,
                                                                dataidxs=None, net_dataidx_map=None)

        # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        # client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    logging.info("data_local_num_dict: %s" % data_local_num_dict)
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


if __name__ == '__main__':
    # data_dir = '/home/datasets/imagenet/ILSVRC2012_dataset'
    data_dir = '/home/datasets/imagenet/imagenet_hdf5/imagenet-shuffled.hdf5'

    client_number = 100
    train_data_num, test_data_num, train_data_global, test_data_global, \
    data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = \
        load_partition_data_ImageNet(None, data_dir,
                                     partition_method=None, partition_alpha=None, client_number=client_number,
                                     batch_size=10)

    print(train_data_num, test_data_num, class_num)
    print(data_local_num_dict)

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
