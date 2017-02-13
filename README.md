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
$ python pre_val_json.py
$ python split_train_val_data.py
```

The python script `pre_train_json.py`, it is used to process the `./data/captions_train2014.json`, it generated a file: `./data/train_images_captions.pkl`, it is a dict which save the captions of each image, like this:
<center>![train_image_captions](https://github.com/chenxinpeng/Optimization-of-image-description-metrics-using-policy-gradient-methods/blob/master/image/1.png)</center>

The script `pre_val_json.py`, it is used to process the `./data/captions_val2014.json`. it generated a file: `./data/val_images_captions.pkl`.

The script `split_train_val_data.py`, because according to the paper, it only use 1665 validation images, the other validation images are used to training. So, I split the validation images into two parts, the 0~1665 images are used to validation, the left are used to training.

### Step 3
Run the scripts:
```bash
$ python create_train_val_all_reference.py
```
and
```bash
$ create_train_val_each_reference.py
```

Let me explain the two scripts, the first script `create_train_val_all_reference.py`, it will generate a JSON file named `train_val_all_reference.json`(about 70M), it saves the ground-truth captions of training and validation images.

The second script `create_train_val_each_reference.py`, it will generate JSON files of every training and validation images. And it saves every JSON file in the folder: `./train_val_reference_json/`

### Step 4
Run the script:
```bash
$ python build_vocab.py
```

This script will build the vocabulary dict. In the data folder, it will generate three files:
 - word_to_idx.pkl
 - idx_to_word.pkl
 - bias_init_vector.npy
  
By the way, I filter the words more than 5 times, you can change this parameter in the script.

### Step 5
In this step, we follow the algorithm in the paper:
<center>![algorithm](https://github.com/chenxinpeng/Optimization-of-image-description-metrics-using-policy-gradient-methods/blob/master/image/2.png)</center>

First, we train the the basic model with MLE(Maximum Likehood Estimation):
```bash
$ CUDA_VISIBLE_DEVICES=0 ipython
>>> import image_caption
>>> image_caption.Train_with_MLE()
```

After training the basic model, you can test and validate the model on test data and validation data:
```bash
>>> image_caption.Test_with_MLE()
>>> image_caption.Val_with_MLE()
```

Second, we train B_phi using MC estimates of Q_theta on a small dataset D(1665 images):
```bash
>>> image_caption.Sample_Q_with_MC()
>>> image_caption.Train_Bphi_Model()
```

After we get the B_phi model, we use RG to optimize the generation:
```bash
>>> image_caption.Train_SGD_update()
```
I have runned several epochs, here I compared the RL results with the no RL results:
![results compared](https://github.com/chenxinpeng/Optimization-of-image-description-metrics-using-policy-gradient-methods/blob/master/image/3.png)

This shows that the policy gradient method is beneficial for image caption.

### COCO evalution
In the `./coco_caption/` folder, we can evaluate the generation results and our each trained model. Please see the python scripts.
