from typing import Optional, List, Any

from torch.utils.data import DataLoader

from datasets.segmentation.abstract_segmentation_dataloader import AbstractSegmentationDataLoader
from datasets.segmentation.cityscapes.dataset import CityscapesSegmentation


class CityscapesSegmentationDataLoader(AbstractSegmentationDataLoader):
    def __init__(
            self,
            data_dir: str,
            train_batch_size: int,
            test_batch_size: int,
            image_size: int,
            data_idxs: Optional[List[int]] = None,
    ) -> None:
        super(CityscapesSegmentationDataLoader, self).__init__(data_dir, train_batch_size, test_batch_size, image_size,
                                                               data_idxs)

        data_transform_args = {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225)
        }

        super(CityscapesSegmentationDataLoader, self)._data_transforms(**data_transform_args)

    def _get_centralized_data_loader(self, data_idxs: Optional[List[int]] = None) -> Any:
        train_dataset = CityscapesSegmentation(self.data_dir, split='train', transform=self.train_transform,
                                               data_idxs=data_idxs)
        val_dataset = CityscapesSegmentation(self.data_dir, split='val', transform=self.val_transform)

        train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                       drop_last=True)
        val_data_loader = DataLoader(dataset=val_dataset, batch_size=self.val_batch_size, shuffle=False, drop_last=True)

        return train_data_loader, val_data_loader, len(train_dataset.classes)

    def _get_partitioned_data_loader(
            self,
            train_data_idxs: Optional[List[int]] = None,
            test_data_idxs: Optional[List[int]] = None,
    ) -> Any:
        train_dataset = CityscapesSegmentation(self.data_dir, split='train', transform=self.train_transform,
                                               data_idxs=train_data_idxs)
        val_dataset = CityscapesSegmentation(self.data_dir, split='val', transform=self.val_transform,
                                             data_idxs=test_data_idxs)

        train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                       drop_last=True)
        val_data_loader = DataLoader(dataset=val_dataset, batch_size=self.val_batch_size, shuffle=False, drop_last=True)

        return train_data_loader, val_data_loader, len(train_dataset.classes)

    def _load_data(self) -> Any:
        train_dataset = CityscapesSegmentation(self.data_dir, split='train', transform=self.train_transform)
        val_dataset = CityscapesSegmentation(self.data_dir, split='val', transform=self.val_transform)

        return (train_dataset.images, train_dataset.targets, train_dataset.classes, val_dataset.images,
                val_dataset.targets, val_dataset.classes)
