import logging
from abc import abstractmethod, ABC
from typing import Any, Optional, List

import numpy as np
from torchvision import transforms

from FedML.fedml_core.non_iid_partition.noniid_partition import non_iid_partition_with_dirichlet_distribution, \
    record_data_stats
from datasets.segmentation import custom_transforms


class AbstractSegmentationDataLoader(ABC):
    def __init__(
            self,
            data_dir: str,
            train_batch_size: int,
            val_batch_size: int,
            image_size: int,
    ) -> None:
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.image_size = image_size,

    def _data_transforms(self, **kwargs) -> Any:
        self.train_transform = transforms.Compose([
            custom_transforms.RandomMirror(),
            custom_transforms.RandomScaleCrop(self.image_size, self.image_size),
            custom_transforms.RandomGaussianBlur(),
            custom_transforms.ToTensor(),
            custom_transforms.Normalize(mean=kwargs['mean'], std=kwargs['std']),
        ])

        self.val_transform = transforms.Compose([
            custom_transforms.FixedScaleCrop(self.image_size),
            custom_transforms.ToTensor(),
            custom_transforms.Normalize(mean=kwargs['mean'], std=kwargs['std']),
        ])

    @abstractmethod
    def _get_centralized_data_loader(self, data_idxs: Optional[List[int]] = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _get_partitioned_data_loader(
            self,
            train_data_idxs: Optional[List[int]] = None,
            test_data_idxs: Optional[List[int]] = None,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _load_data(self) -> Any:
        raise NotImplementedError

    def _partition_data(self, partition: str, n_nets: int, alpha: float) -> Any:
        logging.info('********************* Partitioning data **********************')
        net_data_idx_map = None
        train_images, train_targets, train_categories = self._load_data()[:3]
        n_train = len(train_images)  # Number of training samples

        if partition == 'homo':
            total_num = n_train
            idxs = np.random.permutation(total_num)
            batch_idxs = np.array_split(idxs, n_nets)  # As many splits as n_nets = number of clients
            net_data_idx_map = {i: batch_idxs[i] for i in range(n_nets)}

        # non-iid data distribution
        # TODO: Add custom non-iid distribution option - hetero-fix
        elif partition == 'hetero':
            # This is useful if we allow custom category lists, currently done for consistency
            categories = [train_categories.index(c) for c in train_categories]
            net_data_idx_map = non_iid_partition_with_dirichlet_distribution(train_targets, n_nets, categories, alpha,
                                                                             task='segmentation')

        train_data_cls_counts = record_data_stats(train_targets, net_data_idx_map, task='segmentation')

        return net_data_idx_map, train_data_cls_counts

    def load_partition_data_distributed(
            self,
            process_id: int,
            partition_method: str,
            partition_alpha: float,
            client_number: int,
    ) -> Any:
        net_data_idx_map, train_data_cls_counts = self._partition_data(partition_method, client_number, partition_alpha)

        train_data_num = sum([len(net_data_idx_map[r]) for r in range(client_number)])

        # get global test data
        if process_id == 0:
            train_data_global, test_data_global, class_num = self._get_centralized_data_loader()
            logging.info("Number of global train batches: {} and test batches: {}".format(len(train_data_global),
                                                                                          len(test_data_global)))

            train_data_local_dict = None
            test_data_local_dict = None
            data_local_num_dict = None
        else:
            # get local dataset
            client_id = process_id - 1
            data_idxs = net_data_idx_map[client_id]

            local_data_num = len(data_idxs)
            logging.info("Total number of local images: {} in client ID {}".format(local_data_num, process_id))
            # training batch size = 64; algorithms batch size = 32
            train_data_local, test_data_local, class_num = self._get_centralized_data_loader(data_idxs)
            logging.info(
                "Number of local train batches: {} and test batches: {} in client ID {}".format(len(train_data_local),
                                                                                                len(test_data_local),
                                                                                                process_id))

            data_local_num_dict = {client_id: local_data_num}
            train_data_local_dict = {client_id: train_data_local}
            test_data_local_dict = {client_id: test_data_local}
            train_data_global = None
            test_data_global = None
        return (train_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict,
                test_data_local_dict, class_num)

    def load_partition_data(
            self,
            partition_method: str,
            partition_alpha: float,
            client_number: int,
    ) -> Any:
        net_data_idx_map, train_data_cls_counts = self._partition_data(partition_method, client_number, partition_alpha)

        train_data_num = sum([len(net_data_idx_map[r]) for r in range(client_number)])

        # Global train and test data
        train_data_global, test_data_global, class_num = self._get_centralized_data_loader()
        logging.info("Number of global train batches: {} and test batches: {}".format(len(train_data_global),
                                                                                      len(test_data_global)))

        test_data_num = len(test_data_global)

        # get local dataset
        data_local_num_dict = dict()  # Number of samples for each client
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_idx in range(client_number):
            data_idxs = net_data_idx_map[client_idx]  # get dataId list for client generated using Dirichlet sampling
            local_data_num = len(data_idxs)  # How many samples does client have?
            logging.info("Total number of local images: {} in client ID {}".format(local_data_num, client_idx))

            data_local_num_dict[client_idx] = local_data_num

            # training batch size = 64; algorithms batch size = 32
            train_data_local, test_data_local, class_num = self._get_centralized_data_loader(data_idxs)
            logging.info(
                "Number of local train batches: {} and test batches: {} in client ID {}".format(len(train_data_local),
                                                                                                len(test_data_local),
                                                                                                client_idx))

            # Store data loaders for each client as they contain specific data
            train_data_local_dict[client_idx] = train_data_local
            test_data_local_dict[client_idx] = test_data_local
        return (train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict,
                train_data_local_dict, test_data_local_dict, class_num)
