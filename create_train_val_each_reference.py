# encoding: UTF-8

###############################################################
#
# generate images captions into every json file one by one,
# and get the dict that map the image IDs to image Names
#
###############################################################

import os
import sys
import json
import cPickle as pickle

train_val_imageNames_to_imageIDs = {}
train_imageNames_to_imageIDs = {}
val_imageNames_to_imageIDs = {}

with open('./data/captions_train2014.json') as fr_1:
    train_captions = json.load(fr_1)

for image in train_captions['images']:
    image_name = image['file_name']
    image_id = image['id']
    train_imageNames_to_imageIDs[image_name] = image_id

train_Names_Captions = []
for image in train_captions['annotations']:
    image_id = image['image_id']
    image_caption = image['caption']
    train_Names_Captions.append([image_id, image_caption])

train_count = 0
for imageName, imageID in train_imageNames_to_imageIDs.iteritems():
    print "{},  {},  {}".format(train_count, imageName, imageID)
    train_count += 1

    captions = []
    for item in train_Names_Captions:
        if item[0] == imageID:
            captions.append(item[1])

    json_fw = open('./train_val_reference_json/'+imageName+'.json', 'w')
    json_fw.write('{"info": {"description": "CaptionEval", "url": "https://github.com/chenxinpeng/", "version": "1.0", "year": 2017, "contributor": "Xinpeng Chen", "date_created": "2017.01.26"}, "images": [{"license": 1, "file_name": "' + imageName + '", "id": ' + str(imageID) + '}]')

    json_fw.write(' ,"licenses": [{"url": "test", "id": 1, "name": "test"}], ')
    json_fw.write('"type": "captions", "annotations": [')

    id_count = 0
    for idx, each_sent in enumerate(captions):
        if idx != len(captions)-1:
            if '\n' in each_sent:
                each_sent = each_sent.replace('\n', '')
            if '\\' in each_sent:
                each_sent = each_sent.replace('\\', '')
            if '"' in each_sent:
                each_sent = each_sent.replace('"', '')
            json_fw.write('{"image_id": ' + str(imageID) + ', "id": ' + str(id_count) + ', "caption": "' + each_sent + '"}, ')
        else:
            if '\n' in each_sent:
                each_sent = each_sent.replace('\n', '')
            if '\\' in each_sent:
                each_sent = each_sent.replace('\\', '')
            if '"' in each_sent:
                each_sent = each_sent.replace('"', '')
            json_fw.write('{"image_id": ' + str(imageID) + ', "id": ' + str(id_count) + ', "caption": "' + each_sent + '"}]}')
        id_count += 1
    json_fw.close()

# Validation json file
with open('./data/captions_val2014.json') as fr_2:
    val_captions = json.load(fr_2)

for image in val_captions['images']:
    image_name = image['file_name']
    image_id = image['id']
    val_imageNames_to_imageIDs[image_name] = image_id

val_Names_Captions = []
for image in val_captions['annotations']:
    image_id = image['image_id']
    image_caption = image['caption']
    val_Names_Captions.append([image_id, image_caption])

val_count = 0
for imageName, imageID in val_imageNames_to_imageIDs.iteritems():
    print "{},  {},  {}".format(val_count, imageName, imageID)

    captions = []
    for item in val_Names_Captions:
        if item[0] == imageID:
            captions.append(item[1])

    json_fw = open('./train_val_reference_json/'+imageName+'.json', 'w')
    json_fw.write('{"info": {"description": "CaptionEval", "url": "https://github.com/chenxinpeng/", "version": "1.0", "year": 2017, "contributor": "Xinpeng Chen", "date_created": "2017.01.26"}, "images": [{"license": 1, "file_name": "' + imageName + '", "id": ' + str(imageID) + '}]')

    json_fw.write(' ,"licenses": [{"url": "test", "id": 1, "name": "test"}], ')
    json_fw.write('"type": "captions", "annotations": [')

    id_count = 0
    for idx, each_sent in enumerate(captions):
        if idx != len(captions)-1:
            if '\n' in each_sent:
                each_sent = each_sent.replace('\n', '')
            if '\\' in each_sent:
                each_sent = each_sent.replace('\\', '')
            if '"' in each_sent:
                each_sent = each_sent.replace('"', '')
            json_fw.write('{"image_id": ' + str(imageID) + ', "id": ' + str(id_count) + ', "caption": "' + each_sent + '"}, ')
        else:
            if '\n' in each_sent:
                each_sent = each_sent.replace('\n', '')
            if '\\' in each_sent:
                each_sent = each_sent.replace('\\', '')
            if '"' in each_sent:
                each_sent = each_sent.replace('"', '')
            json_fw.write('{"image_id": ' + str(imageID) + ', "id": ' + str(id_count) + ', "caption": "' + each_sent + '"}]}')
        id_count += 1
    val_count += 1
    json_fw.close()

for k, item in train_imageNames_to_imageIDs.iteritems():
    train_val_imageNames_to_imageIDs[k] = item
for k, item in val_imageNames_to_imageIDs.iteritems():
    train_val_imageNames_to_imageIDs[k] = item

with open('./data/train_val_imageNames_to_imageIDs.pkl', 'w') as fw_2:
    pickle.dump(train_val_imageNames_to_imageIDs, fw_2)


