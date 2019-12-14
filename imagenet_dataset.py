"""
Loads the dataset from image net.
"""
from utils import preprocess_image, create_dataset_from_main_directory
from torch.utils import data
from torch import Tensor
import torch
import random
import os


class ImageNetData(data.Dataset):
    """
    Dataset of various classes from ImageNet
    """
    def __init__(self, data_root: str = "imagenet/", limit=-1):
        self.samples = []
        self.data_root = data_root
        self._init_dataset(limit)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # print("asking for index {} with label {}".format(index, label))
        return self.samples[index]


    def _init_dataset(self, limit: int):
        """
        Populates the samples list with a tuple (path, label)
        The images are not loaded since it would be memory
        inefficient.
        """
        dataset = create_dataset_from_main_directory(self.data_root)
        samples = []
    
        for image_path, label in dataset.items():
            if label is not None:
                samples.append((image_path, label))
        if limit == -1:
            limit = len(samples)
        # Shuffle to avoid overfitting!
        random.shuffle(samples)

        for image, label in samples:
            if len(self.samples) >= limit:
                break
        
            processed_image = preprocess_image(image)
            if torch.cuda.is_available():
                processed_image = processed_image.cuda()
            self.samples.append((processed_image, label))
