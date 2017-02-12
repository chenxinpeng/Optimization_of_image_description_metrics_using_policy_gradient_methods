# Optimization of image description metrics using policy gradient methods
This is Tensorflow implement of paper: [Optimization of image description metrics using policy gradient methods](https://arxiv.org/abs/1612.00370).

## How to run the code
### Step 1: Extract image features
Go into the `./inception` directory, the python script which used to extract features is: `extract_inception_bottleneck_feature.py`.

In this python script, there are few parameters you should modified:
 - `image_path`: the MSCOCO image path, e.g. `/path/to/msococo/train2014`, `/path/to/msococo/val2014`, `/path/to/msococo/test2014`
 - `feats_save_path`: the feature directory which you want to saved.
 - `model_path`: the pre-trained **inception-V3** tensorflow model. And I uploaded this model on the Google Drive: [tensorflow_inception_graph.pb](https://drive.google.com/open?id=0B65vBUruA6N4Y2dtVHBJMVhodjA)
 

After you modified the parameters, we can extract image features, in the terminal:
 ```bash
 $ CUDA_VISIBLE_DEVICES=3 python extract_inception_bottleneck_feature.py
 ```
Also, you can run the code without GPU:
 ```bash
 $ CUDA_VISIBLE_DEVICES="" python extract_inception_bottleneck_feature.py
 ```

In my experiment, I save the `train2014` image feature in the folder: `./inception/train_feats`, `val2014` image feature are saved in the folder: `./inception/val_feats`, and the `test2014` image features are saved in the folder: `test_feats`
And at the same time, I saved the `train2014`+`val2014` image features in the folder: `./inception/train_val_feats`

### Step 2
Run the scripts:
```bash
$ python pre_train_json.py
$ python pre_val_json.py'
$ python split_train_val_data.py
```

The python script `pre_train_json.py`, it is used to process the `./data/captions_train2014.json`, it generated a file: `./data/train_images_captions.pkl`, it is a dict which save the captions of each image, like this:
<center>![train_image_captions](https://github.com/chenxinpeng/Optimization-of-image-description-metrics-using-policy-gradient-methods/blob/master/image/1.png)</center>

The script `pre_val_json.py`, it is used to process the `./data/captions_val2014.json`. it generated a file: `./data/val_images_captions.pkl`.

The script `split_train_val_data.py`, because according to the paper, it only use 1665 validation images, the other validation images are used to training. So, I split the validation images into two parts, the 0~1665 images are used to validation, the left are used to training.

###Step 3


