# Optimization of image description metrics using policy gradient methods
This is Tensorflow implement of paper: [Optimization of image description metrics using policy gradient methods](https://arxiv.org/abs/1612.00370).

## How to run the code
### Extract image features
Go into the `./inception` directory, the python script which used to extract features is: `extract_inception_bottleneck_feature.py`.

In this python script, there are few parameters you should modified:
 - `image_path`: the MSCOCO image path, e.g. `/path/to/msococo/train2014`, `/path/to/msococo/val2014`, /path/to/msococo/test2014
 - `feats_save_path`: the feature directory which you want to saved.
 - `model_path`: the pre-trained **inception-V3** tensorflow model
 

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
