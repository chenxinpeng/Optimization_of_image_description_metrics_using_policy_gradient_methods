#!/usr/bin/env python
# coding=utf-8
import os
import json
import cPickle as pickle

test_results_save_path = '../val2014_results_model_MLP-486.txt'
test_results = open(test_results_save_path).read().splitlines()

images_captions = {}
captions = []
names = []
for idx, item in enumerate(test_results):
    if idx % 2 == 0:
        names.append(item)
    if idx % 2 == 1:
        captions.append(item)

for idx, name in enumerate(names):
    print idx, ' ', name
    images_captions[name] = captions[idx]

with open('../data/val2014_images_ids_to_names.pkl', 'r') as fr_1:
    test2014_images_ids_to_names = pickle.load(fr_1)

names_to_ids = {}
for key, item in test2014_images_ids_to_names.iteritems():
    names_to_ids[item] = key

fw_1 = open('captions_val2014_results.json', 'w')
fw_1.write('[')

for idx, name in enumerate(names):
    print idx, ' ', name
    tmp_idx = names.index(name)
    caption = captions[tmp_idx]
    caption = caption.replace(' ,', ',')
    caption = caption.replace('"', '')
    caption = caption.replace('\n', '')
    if idx != len(names)-1:
        fw_1.write('{"image_id": ' + str(names_to_ids[name]) + ', "caption": "' + str(caption) + '"}, ')
    else:
        fw_1.write('{"image_id": ' + str(names_to_ids[name]) + ', "caption": "' + str(caption) + '"}]')

fw_1.close()
