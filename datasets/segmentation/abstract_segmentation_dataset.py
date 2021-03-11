from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, List

from torch.utils.data import Dataset


class AbstractSegmentationDataset(ABC, Dataset):
    def __init__(
            self,
            root_dir: str,
            split: str,
            transform: Optional[Callable],
            data_idxs: Optional[List[int]],
    ) -> None:
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data_idxs = data_idxs
        self.images = list()
        self.masks = list()
        self.targets = None

        self._preprocess()
        self._generate_targets()

        if data_idxs is not None:
            self.images = [self.images[i] for i in data_idxs]
            self.masks = [self.masks[i] for i in data_idxs]

    @abstractmethod
    def _preprocess(self) -> None:
        """
        Pre-process the dataset to get mask and file paths of the images.

        Raises:
            AssertionError: When length of images and masks differs.
        """
        raise NotImplementedError

    @abstractmethod
    def _generate_targets(self) -> None:
        """
        Used to generate targets which in turn is used to partition data in an non-IID setting.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    @property
    @abstractmethod
    def classes(self) -> Any:
        raise NotImplementedError
