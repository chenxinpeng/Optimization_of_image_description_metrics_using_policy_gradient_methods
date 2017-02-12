# encoding: UTF-8

# accoding the paper: we hold out a small subset of 1,665 validation images
# for hyper-parameter tuning, and use the remaining combined training and
# validation set for training

import os
import cPickle as pickle

train_images_captions_path = './data/train_images_captions.pkl'
val_images_captions_path = './data/val_images_captions.pkl'

with open(train_images_captions_path, 'r') as fr1:
    train_images_captions = pickle.load(fr1)

with open(val_images_captions_path, 'r') as fr2:
    val_images_captions = pickle.load(fr2)

val_images_names = val_images_captions.keys()

# val_images_names[0:1665] for validation
# val_images_names[1665:] for training
val_names_part_one = val_images_names[0:1665]
val_names_part_two = val_images_names[1665:]

# re-save the train_images_captions, val_images_captions
val_images_captions_new = {}
for img in val_names_part_one:
    val_images_captions_new[img] = val_images_captions[img]

for img in val_names_part_two:
    train_images_captions[img] = val_images_captions[img]

with open(train_images_captions_path, 'w') as fw1:
    pickle.dump(train_images_captions, fw1)

with open(val_images_captions_path, 'w') as fw2:
    pickle.dump(val_images_captions_new, fw2)

