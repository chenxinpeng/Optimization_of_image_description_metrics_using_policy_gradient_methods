#! encoding: UTF-8

import os
import ipdb
import glob
import time
import subprocess
import cPickle as pickle
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import eval_image_caption


model_path = "../models"

annFile = "../data/train_val_all_reference.json"
resFile = "captions_val2014_results.json"

# create coco object and cocoRes object
coco = COCO(annFile)

n_epochs = 500
n_epochs += 2

with open("Bleu_1.pkl", "r") as f:
    Bleu_1 = pickle.load(f)

with open("Bleu_2.pkl", "r") as f:
    Bleu_2 = pickle.load(f)

with open("Bleu_3.pkl", "r") as f:
    Bleu_3 = pickle.load(f)

with open("Bleu_4.pkl", "r") as f:
    Bleu_4 = pickle.load(f)

with open("METEOR.pkl", "r") as f:
    METEOR = pickle.load(f)

with open("CIDEr.pkl", "r") as f:
    CIDEr = pickle.load(f)

for idx_model in range(202, n_epochs, 2):
    model_name = os.path.join(model_path, "model_MLP-" + str(idx_model))
    
    start_time = time.time()
    
    # generate the val2014_results.txt
    eval_image_caption.Val_with_MLE(model_name)

    # call the gen_val_json.py with subprocess
    # we will generate the captions_val2014_results.json file
    subprocess.call(["python", "gen_val_json.py"])

    # after generating the captions_val2014_results.json file
    # we call the coco caption evaluation tools
    cocoRes = coco.loadRes(resFile)

    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    cocoEval.evaluate() 

    # print output evaluation scores
    for metric, score in cocoEval.eval.items():
        print '%s: %.3f'%(metric, score)
        if metric == "Bleu_1":
            Bleu_1.append(score)
        if metric == "Bleu_2":
            Bleu_2.append(score)
        if metric == "Bleu_3":
            Bleu_3.append(score)
        if metric == "Bleu_4":
            Bleu_4.append(score)
        if metric == "METEOR":
            METEOR.append(score)
        if metric == "CIDEr":
            CIDEr.append(score)
    # save the scores immediately
    with open("Bleu_1.pkl", "w") as fw1:
        pickle.dump(Bleu_1, fw1)
    with open("Bleu_2.pkl", "w") as fw2:
        pickle.dump(Bleu_2, fw2)
    with open("Bleu_3.pkl", "w") as fw3:
        pickle.dump(Bleu_3, fw3)
    with open("Bleu_4.pkl", "w") as fw4:
        pickle.dump(Bleu_4, fw4)
    with open("METEOR.pkl", "w") as fw5:
        pickle.dump(METEOR, fw5)
    with open("CIDEr.pkl", "w") as fw6:
        pickle.dump(CIDEr, fw6)

    print "Mdoel {} evaluation time cost: {}".format(model_name, time.time()-start_time)

# draw the pictures
plt.plot(range(len(Bleu_1)), Bleu_1, label="Bleu-1", color="g")
plt.plot(range(len(Bleu_2)), Bleu_2, label="Bleu-2", color="r")
plt.plot(range(len(Bleu_3)), Bleu_3, label="Bleu-3", color="b")
plt.plot(range(len(Bleu_4)), Bleu_4, label="Bleu-4", color="m")
plt.plot(range(len(METEOR)), METEOR, label="METEOR", color="k")
plt.plot(range(len(CIDEr)), CIDEr, label="CIDEr", color="y")
plt.grid(True)
plt.legend(loc=2)
plt.show()
plt.savefig("evalution.png") 