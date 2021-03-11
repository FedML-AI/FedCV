import os
from pathlib import Path
from typing import Optional, Callable, List, Any

import numpy as np
import scipy.io as sio
from PIL.Image import Image

from datasets.segmentation.abstract_segmentation_dataset import AbstractSegmentationDataset


class PascalVocAugmentedSegmentation(AbstractSegmentationDataset):
    @property
    def classes(self) -> Any:
        return ('__background__', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'television',
                'train')

    def __init__(
            self,
            root_dir: str = '../../../data/pascal_voc_augmented',
            split: str = 'train',
            transform: Optional[Callable] = None,
            data_idxs: Optional[List[int]] = None,
    ) -> None:
        self.images_dir = Path('{}/dataset/img'.format(root_dir))
        self.masks_dir = Path('{}/dataset/cls'.format(root_dir))
        self.split_file = Path('{}/dataset/{}.txt'.format(root_dir, split))

        super(PascalVocAugmentedSegmentation, self).__init__(root_dir, split, transform, data_idxs)

    def _preprocess(self) -> None:
        with open(self.split_file, 'r') as file_names:
            for file_name in file_names:
                img_path = Path('{}/{}.jpg'.format(self.images_dir, file_name.strip(' \n')))
                mask_path = Path('{}/{}.mat'.format(self.masks_dir, file_name.strip(' \n')))
                assert os.path.isfile(img_path)
                assert os.path.isfile(mask_path)
                self.images.append(img_path)
                self.masks.append(mask_path)
            assert len(self.images) == len(self.masks)

    def _generate_targets(self) -> None:
        targets = list()
        for i in range(len(self.images)):
            mat = sio.loadmat(self.masks[i], mat_dtype=True, squeeze_me=True, struct_as_record=False)
            categories = mat['GTcls'].CategoriesPresent
            if isinstance(categories, np.ndarray):
                categories = np.asarray(list(categories))
            else:
                categories = np.asarray([categories]).astype(np.uint8)
            targets.append(categories)
        self.targets = np.asarray(targets)

    def __getitem__(self, index: int) -> Any:
        img = Image.open(self.images[index]).convert('RGB')
        mat = sio.loadmat(self.masks[index], mat_dtype=True, squeeze_me=True, struct_as_record=False)
        mask = mat['GTcls'].Segmentation
        mask = Image.fromarray(mask)
        sample = {'image': img, 'label': mask}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.images)


if __name__ == '__main__':
    dataset = PascalVocAugmentedSegmentation()
    print('Train Images: {}'.format(len(dataset)))
    assert len(dataset) == 8498

    dataset = PascalVocAugmentedSegmentation(split='val')
    print('Val Images: {}'.format(len(dataset)))
    assert len(dataset) == 2857
