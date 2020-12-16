import numpy as np
import torch.utils.data as data
from torchvision import transforms

from .datasets import PascalVocAugmentedSegmentation
from .transforms import RandomMirror, RandomScaleCrop, RandomGaussianBlur, ToTensor, Normalize, FixedScaleCrop


def _data_transforms_pascal_voc_augmented():
    PASCAL_VOC_MEAN = (0.485, 0.456, 0.406)
    PASCAL_VOC_STD = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        RandomMirror(),
        RandomScaleCrop(513, 513),
        RandomGaussianBlur(),
        ToTensor(),
        Normalize(mean=PASCAL_VOC_MEAN, std=PASCAL_VOC_STD),
    ])

    val_transform = transforms.Compose([
        FixedScaleCrop(513),
        ToTensor(),
        Normalize(mean=PASCAL_VOC_MEAN, std=PASCAL_VOC_STD),
    ])

    return train_transform, val_transform


# for centralized training
def get_dataloader(_, data_dir, train_bs, test_bs, data_idxs=None):
    return get_dataloader_pascal_voc_augmented(data_dir, train_bs, test_bs, data_idxs)


# for local devices
def get_dataloader_test(data_dir, train_bs, test_bs, data_idxs_train, data_idxs_test):
    return get_dataloader_pascal_voc_augmented_test(data_dir, train_bs, test_bs, data_idxs_train, data_idxs_test)


def get_dataloader_pascal_voc_augmented(data_dir, train_bs, test_bs, data_idxs=None):
    transform_train, transform_test = _data_transforms_pascal_voc_augmented()

    train_ds = PascalVocAugmentedSegmentation(data_dir,
                                              split='train',
                                              transform=transform_train,
                                              data_idxs=data_idxs)

    test_ds = PascalVocAugmentedSegmentation(data_dir,
                                             split='val',
                                             transform=transform_test)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl, len(train_ds.classes)


def get_dataloader_pascal_voc_augmented_test(data_dir, train_bs, test_bs, data_idxs_train=None, data_idxs_test=None):
    transform_train, transform_test = _data_transforms_pascal_voc_augmented()

    train_ds = PascalVocAugmentedSegmentation(data_dir,
                                              split='train',
                                              transform=transform_train,
                                              data_idxs=data_idxs_train)

    test_ds = PascalVocAugmentedSegmentation(data_dir,
                                             split='val',
                                             transform=transform_test,
                                             data_idxs=data_idxs_test)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl, len(train_ds.classes)


def record_net_data_stats(y_train, net_data_idx_map):
    net_cls_counts = {}

    for net_i, data_idx in net_data_idx_map.items():
        unq, unq_cnt = np.unique(np.concatenate(y_train[data_idx]), return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts


def load_pascal_voc_augmented_data(data_dir):
    transform_train, transform_test = _data_transforms_pascal_voc_augmented()

    train_ds = PascalVocAugmentedSegmentation(data_dir, split='train', transform=transform_train)
    test_ds = PascalVocAugmentedSegmentation(data_dir, split='val', transform=transform_test)

    return train_ds.images, train_ds.targets, train_ds.classes, test_ds.images, test_ds.targets, test_ds.classes


# Get a partition map for each client
def partition_data(data_dir, partition, n_nets, alpha):
    train_data_cls_counts = None
    net_data_idx_map = None
    train_images, train_targets, train_categories, _, __, ___ = load_pascal_voc_augmented_data(data_dir)
    n_train = len(train_images)  # Number of training samples

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)  # As many splits as n_nets = number of clients
        net_data_idx_map = {i: batch_idxs[i] for i in range(n_nets)}

    # non-iid data distribution
    # TODO: Add custom non-iid distribution option - hetero-fix
    elif partition == "hetero":
        min_size = 0
        # K = train_dataset.num_classes
        categories = train_categories
        N = n_train  # Number of labels/training samples
        net_data_idx_map = {}

        idx_batch = []

        while min_size < 10:
            idx_batch = [[] for _1 in range(n_nets)]  # Create a list of empty lists for clients
            # for each class in the dataset
            # one image may have multiple categories.
            for c in range(len(categories)):
                if c > 0:
                    idx_k = np.asarray([np.any(train_targets[i] == c) and not np.any(
                        train_targets[i][train_targets[i] < c]) for i in
                                        range(len(train_targets))])

                else:
                    idx_k = np.asarray(
                        [np.any(train_targets[i] == c) for i in range(len(train_targets))])

                idx_k = np.where(idx_k)[0]  # Get the indices of images that have category = c
                np.random.shuffle(idx_k)  # Shuffle these indices

                # alpha, parameter for Dirichlet dist, vector containing positive concentration parameters (larger
                # the value more even the distribution)

                # eg. np.random.dirichlet([10, 20, 30]) -> array([0.12926711, 0.37333834, 0.49739455])
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))

                # Balance
                # If client's index list is smaller than num_labels/num_clients, keep sample value for the
                # client as it is, else change it to 0.
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])

                # Normalize across all samples
                proportions = proportions / proportions.sum()

                # eg. For 10 clients, 15 samples -> [0,0,2,2,2,2,14,14,14] -> 9 elements
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                # Split sample indices based on proportions
                # eg. Split [1,2,3,4,5,6,7,8,9,0,12,14,15,16,13] based on index values in proportions
                # eg. np.split(np.asarray([1,2,3,4,5,6,7,8,9,0,12,14,15,16,13]), [0,0,2,2,2,2,14,14,14])
                # -> [array([], dtype=int64),
                #  array([], dtype=int64),
                #  array([1, 2]),
                #  array([], dtype=int64),
                #  array([], dtype=int64),
                #  array([], dtype=int64),
                #  array([ 3,  4,  5,  6,  7,  8,  9,  0, 12, 14, 15, 16]),
                #  array([], dtype=int64),
                #  array([], dtype=int64),
                #  array([13])]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_data_idx_map[j] = idx_batch[j]

        train_data_cls_counts = record_net_data_stats(train_targets, net_data_idx_map)

        # print("Final partition for clients: ")
        # for key in net_dataidx_map:
        #     print('Client: ', key)
        #     print('Number of samples: ', len(net_dataidx_map[key]))

    return net_data_idx_map, train_data_cls_counts


def load_partition_data_distributed_pascal_voc_augmented(process_id, dataset, data_dir, partition_method,
                                                         partition_alpha,
                                                         client_number, batch_size):
    net_data_idx_map, train_data_cls_counts = partition_data(data_dir,
                                                             partition_method,
                                                             client_number,
                                                             partition_alpha)
    train_data_num = sum([len(net_data_idx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size)
        train_data_local_dict = None
        test_data_local_dict = None
        data_local_num_dict = None
    else:
        # get local dataset
        client_id = process_id - 1
        data_idxs = net_data_idx_map[client_id]
        local_data_num = len(data_idxs)
        train_data_local, test_data_local, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                                      data_idxs)

        data_local_num_dict = {client_id: local_data_num}
        train_data_local_dict = {client_id: train_data_local}
        test_data_local_dict = {client_id: test_data_local}
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, \
           test_data_local_dict, class_num


# Called from main_fedseg
def load_partition_data_pascal_voc_augmented(dataset, data_dir, partition_method, partition_alpha, client_number,
                                             batch_size):
    net_data_idx_map, train_data_cls_counts = partition_data(data_dir,
                                                             partition_method,
                                                             client_number,
                                                             partition_alpha)

    train_data_num = sum([len(net_data_idx_map[r]) for r in range(client_number)])

    # Global train and test data
    train_data_global, test_data_global, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size)
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()  # Number of samples for each client
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        data_idxs = net_data_idx_map[client_idx]  # get dataId list for client generated using Dirichlet sampling
        local_data_num = len(data_idxs)  # How many samples does client have?
        data_local_num_dict[client_idx] = local_data_num

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                                      data_idxs)

        # Store dataloaders for each client as they contain specific data
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
           train_data_local_dict, test_data_local_dict, class_num
