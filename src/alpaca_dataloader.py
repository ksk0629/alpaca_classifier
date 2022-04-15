from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision

from preprocessor import get_transform


class AlpacaDataLoader():
    """Alpaca dataloader class"""

    def __init__(self, data_dir: str, resize_size: Tuple[int, int], mean: Tuple[float, float, float], std: Tuple[float, float, float],
                 train_size: float, batch_size: int, shuffle: bool, seed: Optional[int]):
        if seed is not None:
            torch.manual_seed(seed)

        # Load the dataset
        self.__data = torchvision.datasets.ImageFolder(root=data_dir, transform=get_transform(resize_size=resize_size, mean=mean, std=std))
        self.idx_to_class = {i: c for c, i in self.__data.class_to_idx.items()}

        # Split the data into for training and for evaluating
        actual_train_size = int(len(self.__data) * train_size)
        actual_dev_size = len(self.__data) - actual_train_size
        self.__data_train, self.__data_dev = torch.utils.data.random_split(self.__data, [actual_train_size, actual_dev_size])

        # Create the data loaders
        self.loader_train = DataLoader(self.__data_train, batch_size=batch_size, shuffle=shuffle)
        self.loader_dev = DataLoader(self.__data_dev, batch_size=batch_size, shuffle=shuffle)

        # Create iterator for the data loader for training
        self.data_train_iterator = iter(self.loader_train)

    def show_images(self) -> None:
        images, _ = self.data_train_iterator.next()
        
        images_small = torchvision.utils.make_grad(images) / 2 + 0.5
        images_np = images_small.numpy()
        plt.imshow(np.transpose(images_np, (1, 2, 0)))
        plt.show()
