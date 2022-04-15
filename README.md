# alpaca_classifier
This repository is for classifying alpaca images. I check if the codes run without problem on only google colaboratory, whose version of Python is 3.7.12.

## Environment
All you have to do is install mlflow: `pip install mlflow`.

## Quickstart
### Preparation
Download Alpaca Dataset for Image Classification from kaggle: https://www.kaggle.com/datasets/sid4sal/alpaca-dataset-small.

### Training
Run `train.py`: `python ./src/train.py`.

### Classify
Run `classify.sh`: `./classify.py`.

## To change training config
Most of training configs are written in `config.yaml` including a CNN structure.