import json
import math
import os
from tkinter import N
from typing import Dict, Optional

import mlflow
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer():
    """Trainer class"""

    def __init__(self, model: nn.Module, loader_train: DataLoader, loader_dev: DataLoader, idx_to_class: Dict[int, str], seed: Optional[int]) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if seed is not None:
            torch.manual_seed(seed)

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

        self.loader_train = loader_train
        self.loader_dev = loader_dev
        self.idx_to_class = idx_to_class

        self.global_step = 0
        self.loss_history = []
        self.accuracy_history = []

    @property
    def best_loss_epoch(self) -> int:
        best_loss_epoch = np.argmin(self.loss_history) if self.loss_history is not None else 0
        return best_loss_epoch

    @property
    def best_accuracy_epoch(self) -> int:
        best_accuracy_epoch = np.argmax(self.accuracy_history) if self.accuracy_history is not None else 0
        return best_accuracy_epoch

    def train_and_evaluate(self, num_epochs: int, output_dir: str, should_save_best: bool) -> None:
        for epoch in range(1, num_epochs+1):
            print(f"========== epoch: {epoch} ==========")
            self.__train_one_epoch(epoch=epoch)
            self.__evaluate_one_epoch(epoch=epoch)

            if should_save_best:
                self.__save_whilst_running(epoch=epoch, output_dir=output_dir)

        output_model_path = os.path.join(output_dir, "model.pt")
        torch.save(self.model.state_dict(), output_model_path)
        output_json_path = os.path.join(output_dir, f"idx_to_class.json")
        with open(output_json_path, "w") as j_file:
            json.dump(self.idx_to_class, j_file, indent=4)

        mlflow.pytorch.log_model(self.model, "model")

    def __train_one_epoch(self, epoch: int) -> float:
        self.model.train()

        print(f"learning_rate: {self.optimizer.param_groups[0]['lr']}")

        all_losses = []
        correct = 0
        total = 0

        num_steps = len(self.loader_train)
        shown_steps = math.ceil(num_steps / 3)
        shown_steps = 1 if shown_steps == 0 else shown_steps
        for step, (images, labels) in enumerate(self.loader_train, 1):
            self.global_step += 1

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Calculate the loss value
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Calculate the accuracy
            _, predictions = torch.max(outputs.data, 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            loss_value = loss.item()
            mlflow.log_metric("train loss_step", loss_value, step=self.global_step)

            if step % shown_steps == 0 or step == num_steps:
                print(f"step {step}, loss: {loss_value}")
            all_losses.append(loss_value)
        self.scheduler.step()

        loss_value = sum(all_losses) / num_steps
        mlflow.log_metric("train loss_epoch", loss_value, step=epoch)

        accuracy = correct / total
        mlflow.log_metric("train accuracy", accuracy, step=epoch)

        self.loss_history.append(loss_value)

    def __evaluate_one_epoch(self, epoch: int) -> None:
        self.model.eval()

        all_losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            num_steps = len(self.loader_dev)
            for (images, labels) in self.loader_dev:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss_value = loss.item()
                all_losses.append(loss_value)

                _, predictions = torch.max(outputs.data, 1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        loss_value = sum(all_losses) / num_steps
        mlflow.log_metric("eval loss_epoch", loss_value, step=epoch)

        accuracy = correct / total
        mlflow.log_metric("eval accuracy", accuracy, step=epoch)
        print(f"eval accuracy: {accuracy}")
        
        self.accuracy_history.append(accuracy)

    def __save_whilst_running(self, epoch: int, output_dir: str) -> None:
        extension = ".tar"

        if epoch-1 == self.best_loss_epoch:
            output_path = os.path.join(output_dir, f"best_loss{extension}")

            parameters = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "idx_to_class": self.idx_to_class
                }

            torch.save(parameters, output_path)

        if epoch-1 == self.best_accuracy_epoch:
            output_path = os.path.join(output_dir, f"best_accuracy{extension}")

            parameters = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "idx_to_class": self.idx_to_class
                }

            torch.save(parameters, output_path)