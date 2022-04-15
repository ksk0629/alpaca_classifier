#!/bin/sh

model_name=NaiveCNN
model_dir=model/test
image_path_alpaca=data/alpaca/038fae9e70c4c3f1.jpg
image_path_not_alpaca="data/not_alpaca/002bdaf1c177effd.jpg"

python ./src/classify.py -n $model_name -d $model_dir -i $image_path_alpaca

python ./src/classify.py -n $model_name -d $model_dir -i $image_path_not_alpaca