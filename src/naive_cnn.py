from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveCNN(nn.Module):
    """Naive CNN class"""

    def __init__(self, num_classes: int, structure: List[Dict[str, Dict[str, int]]], seed: Optional[int]) -> None:
        super(NaiveCNN,self).__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.num_classes = num_classes

        # Create the network
        self.__create_network(structure=structure)

    def __create_network(self, structure: List[Dict[str, Dict[str, int]]]) -> None:
        self.__block_names = []
        network = []

        for block in structure:
            layer_func, param_dict = self.__analyse(block)
            network.append(layer_func(**param_dict))

            assert len(network) == len(self.__block_names)

        new_network = self.__add_flatten_layers(network)
        self.network = nn.Sequential(*new_network)

    def __analyse(self, block: Dict[str, Dict[str, int]]):
        for block_name, param_dict in block.items():
            self.__block_names.append(block_name)

            if block_name == "conv":
                layer_func = self.__conv_block
            elif block_name == "dense":
                layer_func = self.__dense_block
            elif block_name == "out":
                layer_func = self.__out_block
                param_dict["out_features"] = self.num_classes
            else:
                msg = f"There is no block {block_name}."
                raise ValueError(msg)
            
            return layer_func, param_dict

    def __add_flatten_layers(self, network: object):
        new_network = [network[0]]
        pre_block_name = self.__block_names[0]
        for block_name, block in zip(self.__block_names[1:], network[1:]):
            if pre_block_name != "dense" and (block_name == "dense" or block_name == "out"):
                new_network.append(nn.Flatten())
            new_network.append(block)
            pre_block_name = block_name

        return new_network

    def __conv_block(self, in_channels: int, out_channels: int, kernel_size_conv: int, stride_conv: int, padding: int,
                     kernel_size_pool, stride_pool: int):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size_conv, stride=stride_conv, padding=padding),
            torch.nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_pool)
        )
        return conv_block

    def __dense_block(self, in_features: int, out_features: int):
        dense_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            torch.nn.ReLU(inplace=True)
        )
        return dense_block

    def __out_block(self, in_features: int, out_features: int):
        out_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.Softmax(dim=1)
        )
        return out_block

    def forward(self, x) -> torch.Tensor:
        x = self.network(x)
        return x

    @classmethod
    def load_model(cls, num_classes: int, model_path: str) -> object:
        model = cls(num_classes=num_classes, seed=None)
        model.load_state_dict(torch.load(model_path))
        return model
