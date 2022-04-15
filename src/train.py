import argparse
import os
from typing import Optional, Tuple

import mlflow
import yaml

from alpaca_dataloader import AlpacaDataLoader
import utils
from naive_cnn import NaiveCNN
from trainer import Trainer


def train_and_evaluate(model: NaiveCNN, data_dir: str, resize_size: Tuple[int, int], mean: Tuple[float, float, float], std: Tuple[float, float, float],
                       train_size: float, batch_size: int, shuffle: bool, num_epochs: int, output_dir: str, should_save_best: bool, seed: Optional[int]) -> None:
    alpaca_dataloader = load_alpaca_data(data_dir=data_dir, resize_size=resize_size, mean=mean, std=std, train_size=train_size, batch_size=batch_size, shuffle=shuffle, seed=seed)
    trainer = Trainer(model=model, loader_train=alpaca_dataloader.loader_train, loader_dev=alpaca_dataloader.loader_train, idx_to_class=alpaca_dataloader.idx_to_class, seed=seed)
    
    trainer.train_and_evaluate(num_epochs=num_epochs, output_dir=output_dir, should_save_best=should_save_best)


def load_alpaca_data(data_dir: str, resize_size: Tuple[int, int], mean: Tuple[float, float, float], std: Tuple[float, float, float],
                     train_size: float, batch_size: int, shuffle: bool, seed: Optional[int]) -> AlpacaDataLoader:
    return AlpacaDataLoader(data_dir=data_dir, resize_size=resize_size, mean=mean, std=std, train_size=train_size, batch_size=batch_size, shuffle=shuffle, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate CNN for alpaca dataset.")
    parser.add_argument("-c", "--config_yaml_path", required=False, type=str, default="./config.yaml")
    args = parser.parse_args()

    # Load configs
    with open(args.config_yaml_path, "r") as yaml_f:
        config = yaml.safe_load(yaml_f)
    config_mlflow = config["mlflow"]
    config_common = config["common"]
    config_dataset = config["dataset"]
    config_model = config["model"]
    config_train = config["train"]

    os.makedirs(config_train["output_dir"], exist_ok=True)

    model_class = utils.get_model(config_model.pop("model_name"))
    model = model_class(**config_common, **config_model)
    config_model.pop("structure")

    mlflow.set_experiment(config_mlflow["experiment_name"])
    with mlflow.start_run(run_name=config_mlflow["run_name"]):
        mlflow.log_text(str(model.network), "architecture.txt")
        mlflow.log_params(config_common)
        mlflow.log_params(config_dataset)
        mlflow.log_param("model_class", model_class)
        mlflow.log_params(config_model)
        mlflow.log_params(config_train)

        train_and_evaluate(model=model, **config_common, **config_dataset, **config_train)
