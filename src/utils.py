import json
import os
from typing import Dict, Tuple, Union

import lycon
import numpy as np
import torchvision

from naive_cnn import NaiveCNN

def get_model(model_name: str) -> Union[NaiveCNN, None]:
    if model_name == "NaiveCNN":
        return NaiveCNN
    else:
        msg = f"There is no {model_name}."
        raise ValueError(msg)


def load_model_from_path(model_name: str, model_dir: str) -> Tuple[Union[NaiveCNN, None], Dict[int, str]]:
    json_path = os.path.join(model_dir, "idx_to_class.json")
    with open(json_path, "r") as j_file:
        idx_to_class_str = json.load(j_file)
    idx_to_class = {int(key): value for key, value in idx_to_class_str.items()}

    model_class = get_model(model_name)
    model_path = os.path.join(model_dir, "model.pt")
    model = model_class.load_model(num_classes=len(idx_to_class), model_path=model_path)

    return model, idx_to_class


def read_image_as_pil(image_path: str) -> object:
    image_np = read_image(image_path)
    image_pil = torchvision.transforms.ToPILImage()(image_np)
    return image_pil


def read_image(image_path: str) -> np.array:
    image = lycon.load(image_path)
    return image
