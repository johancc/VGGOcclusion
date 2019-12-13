"""
Loads the dataset from image net.
"""
from utils import preprocess_image, create_dataset_from_main_directory
from torch.utils import data

import os


class ImageNetData(data.Dataset):
    """
    Dataset of various classes from ImageNet
    """
    def __init__(self, data_root: str = "image_net_data/", limit=-1):
        self.samples = []
        self.data_root = data_root
        self._init_dataset(limit)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        return preprocess_image(img_path), label


    def _init_dataset(self, limit: int):
        """
        Populates the samples list with a tuple (path, label)
        The images are not loaded since it would be memory
        inefficient.
        """

        dataset = create_dataset_from_main_directory(self.data_root)
        for image_path, label in dataset.items():
            if label is not None:
                self.samples.append([image_path, label])
                if len(self.samples) >= limit != -1:
                    break
