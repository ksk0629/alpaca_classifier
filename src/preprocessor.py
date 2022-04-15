from typing import Tuple

from torchvision import transforms


def get_transform(resize_size: Tuple[int, int], mean: Tuple[float, float, float], std: Tuple[float, float, float]):
    # Create a preprocess sequence
    transform = transforms.Compose(
            [transforms.Resize(resize_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ])

    return transform