from datasets.segmentation.abstract_segmentation_dataloader import AbstractSegmentationDataLoader
from datasets.segmentation.cityscapes.data_loader import CityscapesSegmentationDataLoader
from datasets.segmentation.pascal_voc_augmented.data_loader import PascalVocAugmentedSegmentationDataLoader


class SegmentationDataLoaderFactory:
    @staticmethod
    def get_data_loader(dataset: str, **kwargs) -> AbstractSegmentationDataLoader:
        if dataset == 'pascal_voc':
            return PascalVocAugmentedSegmentationDataLoader(**kwargs)
        elif dataset == 'cityscapes':
            return CityscapesSegmentationDataLoader(**kwargs)
        else:
            raise NotImplementedError('The dataset "{}" has not been implemented yet'.format(dataset))
