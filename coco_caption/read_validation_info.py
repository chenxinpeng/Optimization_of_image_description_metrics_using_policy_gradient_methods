#!/usr/bin/env python
# coding=utf-8

import os
import json
import cPickle as pickle

image_info_test2014_path = '../data/captions_val2014.json'

image_info_json = json.load(open(image_info_test2014_path, 'r'))

images_info = image_info_json["images"]

imageIds_to_imageNames = {}
for image in images_info:
    id = int(image["id"])
    name = image["file_name"]
    imageIds_to_imageNames[id] = name

with open("val2014_images_ids_to_names.pkl", 'w') as fw_1:
    pickle.dump(imageIds_to_imageNames, fw_1)
