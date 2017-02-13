# encoding: UTF-8

import os
import glob
import json
import cPickle as pickle


train_val_imageNames_to_imageIDs = {}
train_val_Names_Captions = []
#train_imageNames_to_imageIDs = {}
#val_imageNames_to_imageIDs = {}

################################################################
with open('./data/captions_train2014.json') as fr_1:
    train_captions = json.load(fr_1)

for image in train_captions['images']:
    image_name = image['file_name']
    image_id = image['id']
    train_val_imageNames_to_imageIDs[image_name] = image_id

for image in train_captions['annotations']:
    image_id = image['image_id']
    image_caption = image['caption']
    train_val_Names_Captions.append([image_id, image_caption])

#################################################################
with open('./data/captions_val2014.json') as fr_2:
    val_captions = json.load(fr_2)

for image in val_captions['images']:
    image_name = image['file_name']
    image_id = image['id']
    train_val_imageNames_to_imageIDs[image_name] = image_id

for image in val_captions['annotations']:
    image_id = image['image_id']
    image_caption = image['caption']
    train_val_Names_Captions.append([image_id, image_caption])

#################################################################

json_fw = open('./data/train_val_all_reference.json', 'w')
json_fw.write('{"info": {"description": "Test", "url": "https://github.com/chenxinpeng", "version": "1.0", "year": 2017, "contributor": "Chen Xinpeng", "date_created": "2017"}, "images": [')

count = 0
for imageName, imageID in train_val_imageNames_to_imageIDs.iteritems():
    if count != len(train_val_imageNames_to_imageIDs)-1:
        json_fw.write('{"license": 1, "file_name": "' + str(imageName) + '", "id": ' + str(imageID) + '}, ')
    else:
        json_fw.write('{"license": 1, "file_name": "' + str(imageName) + '", "id": ' + str(imageID) + '}]')
    count += 1

json_fw.write(', "licenses": [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1, "name": "Test"}], ')

json_fw.write('"type": "captions", "annotations": [')

flag_count = 0
id_count = 0
for imageName, imageID in train_val_imageNames_to_imageIDs.iteritems():
    print "{},  {},  {}".format(flag_count, imageName, imageID)

    captions = []
    for item in train_val_Names_Captions:
        if item[0] == imageID:
            captions.append(item[1])

    count_captions = 0
    if flag_count != len(train_val_imageNames_to_imageIDs)-1:
        for idx, each_sent in enumerate(captions):
            if '\n' in each_sent:
                each_sent = each_sent.replace('\n', '')
            if '\\' in each_sent:
                each_sent = each_sent.replace('\\', '')
            if '"' in each_sent:
                each_sent = each_sent.replace('"', '')
            json_fw.write('{"image_id": ' + str(imageID) + ', "id": ' + str(id_count) + ', "caption": "' + each_sent + '"}, ')
            id_count += 1

    if flag_count == len(train_val_imageNames_to_imageIDs)-1:
        for idx, each_sent in enumerate(captions):
            if '\n' in each_sent:
                each_sent = each_sent.replace('\n', '')
            if '\\' in each_sent:
                each_sent = each_sent.replace('\\', '')
            if '"' in each_sent:
                each_sent = each_sent.replace('"', '')
            if idx != len(captions)-1:
                json_fw.write('{"image_id": ' + str(imageID) + ', "id": ' + str(id_count) + ', "caption": "' + each_sent + '"}, ')
            else:
                json_fw.write('{"image_id": ' + str(imageID) + ', "id": ' + str(id_count) + ', "caption": "' + each_sent + '"}]}')
            id_count += 1

    flag_count += 1

json_fw.close()
