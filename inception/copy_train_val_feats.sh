#!/bin/bash

DIR="./val_feats/*.npy"

for feat in $DIR
do
    cp $feat ./train_val_feats
done
