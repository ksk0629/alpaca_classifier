import argparse
from typing import Dict

import numpy as np
import torch

import preprocessor
import utils


def classify_from_paths(model_name: str, model_dir: str, image_path: str) -> str:
    image = utils.read_image_as_pil(image_path=image_path)
    transform = preprocessor.get_transform(resize_size=[64, 64], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformed_image = transform(image).unsqueeze(0)
    predicted_class = classify_from_model_path(model_name=model_name, model_dir=model_dir, image=transformed_image)
    
    return predicted_class


def classify_from_model_path(model_name: str, model_dir: str, image: np.array) -> str:
    model, idx_to_class = utils.load_model_from_path(model_name=model_name, model_dir=model_dir)

    predicted_class = classify(model=model, idx_to_class=idx_to_class, image=image)

    return predicted_class


def classify(model: object, idx_to_class: Dict[int, str], image: np.array) -> str:
    output = model(image)
    _, prediction = torch.max(output.data, 1)

    predicted_class = idx_to_class[prediction.item()]

    return predicted_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an image using a trained model.")
    parser.add_argument("-n", "--model_name", required=True, type=str)
    parser.add_argument("-d", "--model_dir", required=True, type=str)
    parser.add_argument("-i", "--image_path", required=True, type=str)
    args = parser.parse_args()

    predicted_class = classify_from_paths(model_name=args.model_name, model_dir=args.model_dir, image_path=args.image_path)

    print(f"{args.image_path} is {predicted_class}.")
