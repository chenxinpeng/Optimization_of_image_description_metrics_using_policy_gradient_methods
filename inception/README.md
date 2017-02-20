Please attetion:

1. First, the original images in MSCOCO have one image, this encoding of this image is PNG, so the extraction will be interrupted. 
   Please use the `check_NOT_JPEG_IMG.sh` file to check out this image and convert it to JPEG file by yourself.

2. Second, when I want to put the training features and validation features into one folder: `train_val_feats`. 
   The number of files is so many that the Linux system can't execute with `cp` command.
   So I use the `copy_train_val_feats.sh` to put the `train_feats`, `val_feats` into one folder: `train_val_feats`.
   
