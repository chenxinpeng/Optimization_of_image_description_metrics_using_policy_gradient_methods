# encoding: UTF-8

import os
import json
import numpy as np
import cPickle as pickle

import time
import ipdb

train_captions_path = './data/captions_val2014.json'
save_images_captions_path = './data/val_images_captions.pkl'

train_captions_fo = open(train_captions_path)
train_captions = json.load(train_captions_fo)

image_ids = []
for annotation in train_captions['annotations']:
    image_ids.append(annotation['image_id'])

# [[filename1, id1], [filename2, id2], ... ]
images_captions = {}
for ii, image in enumerate(train_captions['images']):
    start_time = time.time()

    image_file_name = image['file_name']
    image_id = image['id']
    indices = [i for i, x in enumerate(image_ids) if x == image_id]

    caption = []
    for idx in indices:
        each_cap = train_captions['annotations'][idx]['caption']
        each_cap = each_cap.lower()
        each_cap = each_cap.replace('.', '')
        each_cap = each_cap.replace(',', ' ,')
        each_cap = each_cap.replace('?', ' ?')
        caption.append(each_cap)
    images_captions[image_file_name] = caption
    print "{}  {}  Each image cost: {}".format(ii, image_file_name, time.time()-start_time)

with open(save_images_captions_path, 'w') as fw:
    pickle.dump(images_captions, fw)


