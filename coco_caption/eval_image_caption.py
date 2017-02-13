# encoding: UTF-8

import os
import sys
import glob
import random
import time
import json
from json import encoder
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

import tensorflow as tf

sys.path.append('./coco_caption/')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import ipdb


#############################################################################################################
#
# Step 1: Input: D = {(x^n, y^n): n = 1:N}
# Step 2:Train \Pi(g_{1:T} | x) using MLE on D, MLE: Maximum likehood eatimation
#
############################################################################################################
class CNN_LSTM():
    def __init__(self,
                 n_words,
                 batch_size,
                 feats_dim,
                 project_dim,
                 lstm_size,
                 word_embed_dim,
                 lstm_step,
                 bias_init_vector=None):

        self.n_words = n_words
        self.batch_size = batch_size
        self.feats_dim = feats_dim
        self.project_dim = project_dim
        self.lstm_size = lstm_size
        self.word_embed_dim = word_embed_dim
        self.lstm_step = lstm_step

        # project the image feature vector of dimension 2048 to 512 dimension, with a linear layer
        # self.encode_img_W: 2048 x 512
        # self.encode_img_b: 512
        self.encode_img_W = tf.Variable(tf.random_uniform([feats_dim, project_dim], -0.1, 0.1), name="encode_img_W")
        self.encode_img_b = tf.zeros([project_dim], name="encode_img_b")

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, word_embed_dim], -0.1, 0.1), name="Wemb")

        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)

        self.embed_word_W = tf.Variable(tf.random_uniform([lstm_size, n_words], -0.1, 0.1), name="embed_word_W")

        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name="embed_word_b")
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name="embed_word_b")

        self.baseline_MLP_W = tf.Variable(tf.random_uniform([lstm_size, 1], -0.1, 0.1), name="baseline_MLP_W")
        self.baseline_MLP_b = tf.Variable(tf.zeros([1]), name="baseline_MLP_b")

    ############################################################################################################
    #
    # Class function for step 2
    #
    ############################################################################################################
    def build_model(self):
        images = tf.placeholder(tf.float32, [self.batch_size, self.feats_dim])
        sentences = tf.placeholder(tf.int32, [self.batch_size, self.lstm_step])
        masks = tf.placeholder(tf.float32, [self.batch_size, self.lstm_step])

        images_embed = tf.matmul(images, self.encode_img_W) + self.encode_img_b

        state = self.lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        loss = 0.0
        with tf.variable_scope("LSTM"):
            for i in range(0, self.lstm_step):
                if i == 0:
                    current_emb = images_embed
                else:
                    with tf.device("/cpu:0"):
                        current_emb = tf.nn.embedding_lookup(self.Wemb, sentences[:, i-1])

                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                output, state = self.lstm(current_emb, state)

                if i > 0:
                    labels = tf.expand_dims(sentences[:, i], 1)
                    indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                    concated = tf.concat(1, [indices, labels])
                    onehot_labels = tf.sparse_to_dense( concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)

                    logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
                    cross_entropy = cross_entropy * masks[:, i]
                    current_loss = tf.reduce_sum(cross_entropy)/self.batch_size

                    loss = loss + current_loss
        return loss, images, sentences, masks

    def generate_model(self):
        images = tf.placeholder(tf.float32, [1, self.feats_dim])
        images_embed = tf.matmul(images, self.encode_img_W) + self.encode_img_b

        state = self.lstm.zero_state(batch_size=1, dtype=tf.float32)
        sentences = []

        with tf.variable_scope("LSTM"):
            output, state = self.lstm(images_embed, state)

            with tf.device("/cpu:0"):
                current_emb = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

            for i in range(0, self.lstm_step):
                tf.get_variable_scope().reuse_variables()

                output, state = self.lstm(current_emb, state)

                logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b
                max_prob_word = tf.argmax(logit_words, 1)[0]

                with tf.device("/cpu:0"):
                    current_emb = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                    current_emb = tf.expand_dims(current_emb, 0)
                sentences.append(max_prob_word)

        return images, sentences


##############################################################################
#
# set parameters and path
#
##############################################################################
batch_size = 100
feats_dim = 2048
project_dim = 512
lstm_size = 512
word_embed_dim = 512
lstm_step = 20

idx_to_word_path = '../data/idx_to_word.pkl'
word_to_idx_path = '../data/word_to_idx.pkl'
bias_init_vector_path = '../data/bias_init_vector.npy'

with open(idx_to_word_path, 'r') as fr_3:
    idx_to_word = pickle.load(fr_3)

with open(word_to_idx_path, 'r') as fr_4:
    word_to_idx = pickle.load(fr_4)

bias_init_vector = np.load(bias_init_vector_path)


##########################################################################################
#
# I move the generation model part out of the Val_with_MLE function
#
##########################################################################################
n_words = len(idx_to_word)

val_feats_path = '../inception/val_feats'
val_feats_names = glob.glob(val_feats_path + '/*.npy')
val_images_names = map(lambda x: os.path.basename(x)[0:-4], val_feats_names)

model = CNN_LSTM(n_words = n_words,
                 batch_size = batch_size,
                 feats_dim = feats_dim,
                 project_dim = project_dim,
                 lstm_size = lstm_size,
                 word_embed_dim = word_embed_dim,
                 lstm_step = lstm_step,
                 bias_init_vector = None)
tf_images, tf_sentences = model.generate_model()

def Val_with_MLE(model_path):
    '''
    n_words = len(idx_to_word)

    # version 1: test all validation images
    val_feats_path = '../inception/val_feats'
    val_feats_names = glob.glob(val_feats_path + '/*.npy')
    val_images_names = map(lambda x: os.path.basename(x)[0:-4], val_feats_names)

    model = CNN_LSTM(n_words = n_words,
                     batch_size = batch_size,
                     feats_dim = feats_dim,
                     project_dim = project_dim,
                     lstm_size = lstm_size,
                     word_embed_dim = word_embed_dim,
                     lstm_step = lstm_step,
                     bias_init_vector = None)
    tf_images, tf_sentences = model.generate_model()
    '''
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    fw_1 = open("val2014_results.txt", 'w')
    for idx, img_name in enumerate(val_images_names[0:5000]):
        print "{},  {}".format(idx, img_name)
        start_time = time.time()

        current_feats = np.load( os.path.join(val_feats_path, img_name+'.npy') )
        current_feats = np.reshape(current_feats, [1, feats_dim])

        sentences_index = sess.run(tf_sentences, feed_dict={tf_images: current_feats})
        sentences = []
        for idx_word in sentences_index:
            word = idx_to_word[idx_word]
            word = word.replace('\n', '')
            word = word.replace('\\', '')
            word = word.replace('"', '')
            sentences.append(word)

        punctuation = np.argmax(np.array(sentences) == '<eos>') + 1
        sentences = sentences[:punctuation]
        generated_sentence = ' '.join(sentences)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')

        print generated_sentence,'\n'
        fw_1.write(img_name + '\n')
        fw_1.write(generated_sentence + '\n')
    fw_1.close()

