import os
import os.path
import logging
import numpy as np

from PIL import Image
import torch.utils.data as data
from timm.data import Dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset

from FedML.fedml_core.non_iid_partition.noniid_partition import record_data_stats, \
    non_iid_partition_with_dirichlet_distribution


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx



def make_dataset(dir, class_to_idx, extensions, num_classes=1000):
    images = []

    data_local_num_dict = dict()
    net_dataidx_map = dict()
    sum_temp = 0
    dir = os.path.expanduser(dir)

    i_target = 0 
    for target in sorted(os.listdir(dir)):
        if not (i_target < num_classes):
            break
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        target_num = 0
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    target_num += 1

        net_dataidx_map[class_to_idx[target]] = (sum_temp, sum_temp + target_num)
        data_local_num_dict[class_to_idx[target]] = target_num
        sum_temp += target_num
        i_target += 1

    assert len(images) == sum_temp
    return images, data_local_num_dict, net_dataidx_map



def make_dataset_with_dirichlet_sampling(
    dir, class_to_idx, extensions, client_num, num_classes=1000, alpha=0):
    assert alpha > 0
    images = []

    data_local_num_dict = dict()
    net_dataidx_map = dict()
    sum_temp = 0
    dir = os.path.expanduser(dir)

    i_target = 0 
    label_list = []     # Used for dirichlet sampling
    for target in sorted(os.listdir(dir)):
        if not (i_target < num_classes):
            break
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        target_num = 0
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    label_list.append(i_target)
                    target_num += 1
        net_dataidx_map[class_to_idx[target]] = (sum_temp, sum_temp + target_num)
        data_local_num_dict[class_to_idx[target]] = target_num
        sum_temp += target_num
        i_target += 1

    label_list = np.array(label_list)
    net_dataidx_map = non_iid_partition_with_dirichlet_distribution(
        label_list=label_list, client_num=client_num, classes=num_classes, alpha=alpha)

    return images, data_local_num_dict, net_dataidx_map 



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        pass


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageNet(data.Dataset):

    def __init__(self, data_dir, dataidxs=None, train=True,
                 transform=None, target_transform=None, download=False, client_num=100, alpha=None):
        """
            Generating this class too many times will be time-consuming.
            So it will be better calling this once and put it into ImageNet_truncated.
        """
        self.dataidxs = dataidxs
        self.client_num = client_num
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.loader = default_loader
        if self.train:
            self.data_dir = os.path.join(data_dir, 'train')
        else:
            self.data_dir = os.path.join(data_dir, 'val')

        self.alpha = alpha
        self.all_data, self.data_local_num_dict, self.net_dataidx_map = self.__getdatasets__()
        self.initial_local_data()

    def initial_local_data(self):
        if self.dataidxs == None:
            self.local_data = self.all_data
        elif type(self.dataidxs) == int:
            if self.alpha is not None:
                self.local_data = self.all_data[self.net_dataidx_map[self.dataidxs]]
            else:
                (begin, end) = self.net_dataidx_map[self.dataidxs]
                self.local_data = self.all_data[begin: end]
        else:
            # This is only suitable when not do dirichlet sampling
            assert self.alpha is None
            self.local_data = []
            for idxs in self.dataidxs:
                (begin, end) = self.net_dataidx_map[idxs]
                self.local_data += self.all_data[begin: end]

    def get_local_data(self):
        return self.local_data

    def get_net_dataidx_map(self):
        return self.net_dataidx_map

    def get_data_local_num_dict(self):
        return self.data_local_num_dict

    def __getdatasets__(self):
        # all_data = datasets.ImageFolder(data_dir, self.transform, self.target_transform)

        classes, class_to_idx = find_classes(self.data_dir)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        if self.alpha is not None:
            all_data, data_local_num_dict, net_dataidx_map = make_dataset_with_dirichlet_sampling(
                self.data_dir, class_to_idx, IMG_EXTENSIONS, self.client_num, num_classes=1000, alpha=self.alpha
            )
        else:
            all_data, data_local_num_dict, net_dataidx_map = make_dataset(self.data_dir, class_to_idx, IMG_EXTENSIONS)
        if len(all_data) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.data_dir + "\n"
                                                                                     "Supported extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        return all_data, data_local_num_dict, net_dataidx_map

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], self.target[index]

        path, target = self.local_data[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.local_data)


class ImageNet100(data.Dataset):

    def __init__(self, data_dir, dataidxs=None, train=True,
                 transform=None, target_transform=None, download=False, client_num=100, alpha=None):
        """
            Generating this class too many times will be time-consuming.
            So it will be better calling this once and put it into ImageNet_truncated.
        """
        self.dataidxs = dataidxs
        self.client_num = client_num
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.loader = default_loader
        if self.train:
            self.data_dir = os.path.join(data_dir, 'train')
        else:
            self.data_dir = os.path.join(data_dir, 'val')

        self.alpha = alpha
        self.all_data, self.data_local_num_dict, self.net_dataidx_map = self.__getdatasets__()
        self.initial_local_data()


    def initial_local_data(self):
        if self.dataidxs == None:
            self.local_data = self.all_data
        elif type(self.dataidxs) == int:
            if self.alpha is not None:
                self.local_data = self.all_data[self.net_dataidx_map[self.dataidxs]]
            else:
                (begin, end) = self.net_dataidx_map[self.dataidxs]
                self.local_data = self.all_data[begin: end]
        else:
            # This is only suitable when not do dirichlet sampling
            assert self.alpha is None
            self.local_data = []
            for idxs in self.dataidxs:
                (begin, end) = self.net_dataidx_map[idxs]
                self.local_data += self.all_data[begin: end]

    def get_local_data(self):
        return self.local_data

    def get_net_dataidx_map(self):
        return self.net_dataidx_map

    def get_data_local_num_dict(self):
        return self.data_local_num_dict

    def __getdatasets__(self):
        # all_data = datasets.ImageFolder(data_dir, self.transform, self.target_transform)

        classes, class_to_idx = find_classes(self.data_dir)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        if self.alpha is not None:
            all_data, data_local_num_dict, net_dataidx_map = make_dataset_with_dirichlet_sampling(
                self.data_dir, class_to_idx, IMG_EXTENSIONS, self.client_num, num_classes=100, alpha=self.alpha
            )
        else:
            all_data, data_local_num_dict, net_dataidx_map = make_dataset(
                self.data_dir, class_to_idx, IMG_EXTENSIONS, num_classes=100)
        if len(all_data) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.data_dir + "\n"
                                                                                     "Supported extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        return all_data, data_local_num_dict, net_dataidx_map

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], self.target[index]

        path, target = self.local_data[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.local_data)


class ImageNet_truncated(data.Dataset):

    def __init__(self, imagenet_dataset: ImageNet, dataidxs, net_dataidx_map, train=True, transform=None,
                 target_transform=None, download=False, client_num=100, alpha=None):

        self.dataidxs = dataidxs
        self.client_num = client_num
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.net_dataidx_map = net_dataidx_map
        self.loader = default_loader
        self.all_data = imagenet_dataset.get_local_data()
        self.alpha = alpha
        self.initial_local_data()


    def initial_local_data(self):
        if self.dataidxs == None:
            self.local_data = self.all_data
        elif type(self.dataidxs) == int:
            if self.alpha is not None:
                self.local_data = self.all_data[self.net_dataidx_map[self.dataidxs]]
            else:
                (begin, end) = self.net_dataidx_map[self.dataidxs]
                self.local_data = self.all_data[begin: end]
        elif type(self.dataidxs) == float:
            (begin_origin, end_origin) = self.net_dataidx_map[int(self.dataidxs // 10)]
            # begin = begin_origin + int((end_origin - begin_origin) \
            #     * (self.dataidxs - self.dataidxs // 10))
            # end = begin_origin + int((end_origin - begin_origin) \
            #     * (self.dataidxs - self.dataidxs // 10 + 0.1))
            begin = begin_origin + int((end_origin - begin_origin) \
                * (self.dataidxs - int(self.dataidxs)))
            end = begin_origin + int((end_origin - begin_origin) \
                * (self.dataidxs - int(self.dataidxs) + 0.1))
            logging.info("Get sub dataset of one class into clients: begin: {}, end: {}".format(
                begin, end
            ))
            self.local_data = self.all_data[begin: end]
        else:
            # This is only suitable when not do dirichlet sampling
            assert self.alpha is None
            self.local_data = []
            for idxs in self.dataidxs:
                (begin, end) = self.net_dataidx_map[idxs]
                self.local_data += self.all_data[begin: end]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], self.target[index]

        path, target = self.local_data[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.local_data)
