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

sys.path.append('../')
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

        # At the beginning, I used two layers of MLP, but I think it's wrong
        #self.baseline_MLP2_W = tf.Variable(tf.random_uniform([lstm_size, 1], -0.1, 0.1), name="baseline_MLP2_W")
        #self.baseline_MLP2_b = tf.Variable(tf.zeros([1]), name="baseline_MLP2_b")

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

    ####################################################################################
    #
    # Class function for step 3
    #
    ####################################################################################
    def train_Bphi_model(self):
        encode_img_W = tf.stop_gradient(self.encode_img_W)
        encode_img_b = tf.stop_gradient(self.encode_img_b)
        Wemb = tf.stop_gradient(self.Wemb)

        images = tf.placeholder(tf.float32, [1, self.feats_dim])
        images_embed = tf.matmul(images, encode_img_W) + encode_img_b

        Q_Bleu_1 = tf.placeholder(tf.float32, [1, self.lstm_step])
        Q_Bleu_2 = tf.placeholder(tf.float32, [1, self.lstm_step])
        Q_Bleu_3 = tf.placeholder(tf.float32, [1, self.lstm_step])
        Q_Bleu_4 = tf.placeholder(tf.float32, [1, self.lstm_step])

        weight_Bleu_1 = 0.5
        weight_Bleu_2 = 0.5
        weight_Bleu_3 = 1.0
        weight_Bleu_4 = 1.0

        state = self.lstm.zero_state(batch_size=1, dtype=tf.float32)

        # To avoid creating a feedback loop, we do not back-propagate
        # gradients through the hidden state from this loss
        c, h = state[0], state[1]
        c, h = tf.stop_gradient(c), tf.stop_gradient(h)
        state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

        loss = 0.0

        with tf.variable_scope("LSTM"):
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(Wemb, tf.ones([1], dtype=tf.int64))

            output, state = self.lstm(images_embed, state)
            c, h = state[0], state[1]
            c, h = tf.stop_gradient(c), tf.stop_gradient(h)
            state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

            for i in range(0, self.lstm_step):
                tf.get_variable_scope().reuse_variables()

                output, state = self.lstm(current_embed, state)
                c, h = state[0], state[1]
                c, h = tf.stop_gradient(c), tf.stop_gradient(h)
                state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

                # In our experiments, the baseline estimator is an MLP which takes as input the hidden state of the RNN at step t
                # To avoid creating a feedback loop, we do not back-propagate gradients through the hidden state from this loss
                #if i >= 1:
                baseline_estimator = tf.nn.relu(tf.matmul(state[1], self.baseline_MLP_W) + self.baseline_MLP_b)
                Q_current = weight_Bleu_1 * Q_Bleu_1[:, i] + weight_Bleu_2 * Q_Bleu_2[:, i] + \
                            weight_Bleu_3 * Q_Bleu_3[:, i] + weight_Bleu_4 * Q_Bleu_4[:, i]

                # Equation (8) in the paper
                loss = loss + tf.square(Q_current - baseline_estimator)

        return images, Q_Bleu_1, Q_Bleu_2, Q_Bleu_3, Q_Bleu_4, loss

    def Monte_Carlo_Rollout(self):
        images = tf.placeholder(tf.float32, [1, self.feats_dim])
        images_embed = tf.matmul(images, self.encode_img_W) + self.encode_img_b

        state = self.lstm.zero_state(batch_size=1, dtype=tf.float32)

        gen_sentences = []
        all_sample_sentences = []

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
                gen_sentences.append(max_prob_word)

                if i < self.lstm_step-1:
                    num_sample = self.lstm_step - 1 - i
                    sample_sentences = []
                    for idx_sample in range(num_sample):
                        sample = tf.multinomial(logit_words, 3)
                        sample_sentences.append(sample[0])
                    all_sample_sentences.append(sample_sentences)

        return images, gen_sentences, all_sample_sentences

    ########################################################################
    #
    # Class function for step 4
    #
    ########################################################################
    def Monte_Carlo_and_Baseline(self):
        images = tf.placeholder(tf.float32, [self.batch_size, self.feats_dim])
        images_embed = tf.matmul(images, self.encode_img_W) + self.encode_img_b

        state = self.lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        gen_sentences = []
        all_sample_sentences = []
        all_baselines = []

        with tf.variable_scope("LSTM"):
            output, state = self.lstm(images_embed, state)
            with tf.device("/cpu:0"):
                current_emb = tf.nn.embedding_lookup(self.Wemb, tf.ones([self.batch_size], dtype=tf.int64))

            for i in range(0, self.lstm_step):
                tf.get_variable_scope().reuse_variables()

                output, state = self.lstm(current_emb, state)
                logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b
                max_prob_word = tf.argmax(logit_words, 1)
                with tf.device("/cpu:0"):
                    current_emb = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                    #current_emb = tf.expand_dims(current_emb, 0)
                gen_sentences.append(max_prob_word)

                # compute Q for gt with K Monte Carlo rollouts
                if i < self.lstm_step-1:
                    num_sample = self.lstm_step - 1 - i
                    sample_sentences = []
                    for idx_sample in range(num_sample):
                        sample = tf.multinomial(logit_words, 3)
                        sample_sentences.append(sample)
                    all_sample_sentences.append(sample_sentences)
                # compute eatimated baseline
                baseline = tf.nn.relu(tf.matmul(state[1], self.baseline_MLP_W) + self.baseline_MLP_b)
                all_baselines.append(baseline)

        return images, gen_sentences, all_sample_sentences, all_baselines

    def SGD_update(self, batch_num_images=1000):
        images = tf.placeholder(tf.float32, [batch_num_images, self.feats_dim])
        images_embed = tf.matmul(images, self.encode_img_W) + self.encode_img_b

        Q_rewards = tf.placeholder(tf.float32, [batch_num_images, self.lstm_step])
        Baselines = tf.placeholder(tf.float32, [batch_num_images, self.lstm_step])

        state = self.lstm.zero_state(batch_size=batch_num_images, dtype=tf.float32)

        loss = 0.0

        with tf.variable_scope("LSTM"):
            tf.get_variable_scope().reuse_variables()
            output, state = self.lstm(images_embed, state)

            with tf.device("/cpu:0"):
                current_emb = tf.nn.embedding_lookup(self.Wemb, tf.ones([batch_num_images], dtype=tf.int64))

            for i in range(0, self.lstm_step):
                output, state = self.lstm(current_emb, state)

                logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b
                logit_words_softmax = tf.nn.softmax(logit_words)
                max_prob_word = tf.argmax(logit_words_softmax, 1)
                max_prob = tf.reduce_max(logit_words_softmax, 1)

                current_rewards = Q_rewards[:, i] - Baselines[:, i]
                
                loss = loss + tf.reduce_sum(-tf.log(max_prob) * current_rewards)
                
                with tf.device("/cpu:0"):
                    current_emb = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                    #current_emb = tf.expand_dims(current_emb, 0)

        return images, Q_rewards, Baselines, loss, max_prob, current_rewards, logit_words


##############################################################################
#
# Step 1: set parameters and path
#
##############################################################################
batch_size = 100
feats_dim = 2048
project_dim = 512
lstm_size = 512
word_embed_dim = 512
lstm_step = 30

n_epochs = 500
learning_rate = 0.0001

# Features directory of training and validation images, and the other path
train_val_feats_path = './inception/train_val_feats'
val_feats_path = './inception/val_feats'

loss_images_save_path = './loss_imgs'
loss_file_save_path = 'loss.txt'
model_path = './models'

train_images_captions_path = './data/train_images_captions.pkl'
val_images_captions_path = './data/val_images_captions.pkl'

idx_to_word_path = './data/idx_to_word.pkl'
word_to_idx_path = './data/word_to_idx.pkl'
bias_init_vector_path = './data/bias_init_vector.npy'

# Load pre-processed data
with open(train_images_captions_path, 'r') as fr_1:
    train_images_captions = pickle.load(fr_1)

with open(val_images_captions_path, 'r') as fr_2:
    val_images_captions = pickle.load(fr_2)

with open(idx_to_word_path, 'r') as fr_3:
    idx_to_word = pickle.load(fr_3)

with open(word_to_idx_path, 'r') as fr_4:
    word_to_idx = pickle.load(fr_4)

bias_init_vector = np.load(bias_init_vector_path)


##########################################################################
#
# Step 2: Train, validation and test stage using MLE on Dataset
#
##########################################################################
def Train_with_MLE():
    n_words = len(idx_to_word)
    train_images_names = train_images_captions.keys()

    # change the word of each image captions to index by word_to_idx
    train_images_captions_index = {}
    for each_img, sents in train_images_captions.iteritems():
        sents_index = np.zeros([len(sents), lstm_step], dtype=np.int32)

        for idy, sent in enumerate(sents):
            sent = '<bos> ' + sent + ' <eos>'
            tmp_sent = sent.split(' ')
            tmp_sent = filter(None, tmp_sent)

            for idx, word in enumerate(tmp_sent):
                if idx == lstm_step-1:
                    sents_index[idy, idx] = word_to_idx['<eos>']
                    break
                elif word in word_to_idx:
                    sents_index[idy, idx] = word_to_idx[word]
        train_images_captions_index[each_img] = sents_index
    with open('./data/train_images_captions_index.pkl', 'w') as fw_1:
        pickle.dump(train_images_captions_index, fw_1)

    model = CNN_LSTM(n_words = n_words,
                     batch_size = batch_size,
                     feats_dim = feats_dim,
                     project_dim = project_dim,
                     lstm_size = lstm_size,
                     word_embed_dim = word_embed_dim,
                     lstm_step = lstm_step,
                     bias_init_vector = bias_init_vector)

    tf_loss, tf_images, tf_sentences, tf_masks = model.build_model()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=500, write_version=1)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.initialize_all_variables().run()

    # when you want to train the model from the front model
    #new_saver = tf.train.Saver(max_to_keep=500)
    #new_saver = tf.train.import_meta_graph('./models/model-78.meta')
    #new_saver.restore(sess, tf.train.latest_checkpoint('./models/'))

    loss_fw = open(loss_file_save_path, 'w')
    loss_to_draw = []
    for epoch in range(0, n_epochs):
        loss_to_draw_epoch = []
        # disorder the training images
        random.shuffle(train_images_names)

        for start, end in zip(range(0, len(train_images_names), batch_size),
                              range(batch_size, len(train_images_names), batch_size)):
            start_time = time.time()

            # current_feats: get the [start:end] features
            # current_captions: convert the word to the idx by the word_to_idx
            # current_masks: set the <pad> to zero, the other place to non-zero
            current_feats = []
            current_captions = []

            img_names = train_images_names[start:end]
            for each_img_name in img_names:
                # load this image's feats from the train_val_feats directory
                #each_img_name = each_img_name + '.npy'
                img_feat = np.load( os.path.join(train_val_feats_path, each_img_name+'.npy') )
                current_feats.append(img_feat)

                img_caption_length = len(train_images_captions[each_img_name])
                random_choice_index = random.randint(0, img_caption_length-1)
                img_caption = train_images_captions_index[each_img_name][random_choice_index]
                current_captions.append(img_caption)

            current_feats = np.asarray(current_feats)
            current_captions = np.asarray(current_captions)

            current_masks = np.zeros( (current_captions.shape[0], current_captions.shape[1]), dtype=np.int32 )
            nonzeros = np.array( map(lambda x: (x != 0).sum(), current_captions) )

            for ind, row in enumerate(current_masks):
                row[:nonzeros[ind]] = 1

            _, loss_val = sess.run(
                    [train_op, tf_loss],
                    feed_dict = {
                        tf_images: current_feats,
                        tf_sentences: current_captions,
                        tf_masks: current_masks
                        })
            loss_to_draw_epoch.append(loss_val)

            print "idx: {}  epoch: {}  loss: {}  Time cost: {}".format(start, epoch, loss_val, time.time()-start_time)
            loss_fw.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')

        # draw loss curve every epoch
        loss_to_draw.append(np.mean(loss_to_draw_epoch))
        plt_save_img_name = str(epoch) + '.png'
        plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
        plt.grid(True)
        plt.savefig(os.path.join(loss_images_save_path, plt_save_img_name))

        if np.mod(epoch, 2) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            saver.save(sess, os.path.join(model_path, 'model_MLP'), global_step=epoch)
    loss_fw.close()


def Test_with_MLE():
    model_path = os.path.join('./models', 'model_MLP-486')
    n_words = len(idx_to_word)

    test_feats_path = './inception/test_feats'
    test_feats_names = glob.glob(test_feats_path + '/*.npy')
    test_images_names = map(lambda x: os.path.basename(x)[0:-4], test_feats_names)

    model = CNN_LSTM(n_words = n_words,
                     batch_size = batch_size,
                     feats_dim = feats_dim,
                     project_dim = project_dim,
                     lstm_size = lstm_size,
                     word_embed_dim = word_embed_dim,
                     lstm_step = lstm_step,
                     bias_init_vector = None)

    tf_images, tf_sentences = model.generate_model()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    fw_1 = open("test2014_results_model-486.txt", 'w')
    for idx, img_name in enumerate(test_images_names):
        t0 = time.time()

        current_feats = np.load( os.path.join(test_feats_path, img_name+'.npy') )
        current_feats = np.reshape(current_feats, [1, feats_dim])

        sentences_index = sess.run(tf_sentences, feed_dict={tf_images: current_feats})

        #sentences = map(lambda x: idx_to_word[x], sentences_index)
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

        print "{},  {},  Time cost: {}".format(idx, img_name, time.time()-t0)

    fw_1.close()


def Val_with_MLE():
    model_path = os.path.join('./models', 'model_MLP-486')
    n_words = len(idx_to_word)

    # version 1: test all validation images
    val_feats_path = './inception/val_feats'
    val_feats_names = glob.glob(val_feats_path + '/*.npy')
    val_images_names = map(lambda x: os.path.basename(x)[0:-4], val_feats_names)

    # version 2: test only in the 1665 validation images
    #val_feats_path = './inception/val_feats_v2'
    #with open('./data/val_images_captions.pkl', 'r') as fr_1:
    #    val_images_names = pickle.load(fr_1).keys()

    model = CNN_LSTM(n_words = n_words,
                     batch_size = batch_size,
                     feats_dim = feats_dim,
                     project_dim = project_dim,
                     lstm_size = lstm_size,
                     word_embed_dim = word_embed_dim,
                     lstm_step = lstm_step,
                     bias_init_vector = None)
    tf_images, tf_sentences = model.generate_model()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    fw_1 = open("val2014_results_model_MLP-486.txt", 'w')
    for idx, img_name in enumerate(val_images_names):
        print "{},  {}".format(idx, img_name)
        start_time = time.time()

        current_feats = np.load( os.path.join(val_feats_path, img_name+'.npy') )
        current_feats = np.reshape(current_feats, [1, feats_dim])

        sentences_index = sess.run(tf_sentences, feed_dict={tf_images: current_feats})
        #sentences = map(lambda x: idx_to_word[x], sentences_index)
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


##########################################################################################################
#
# Step 3: Train B_phi using MC estimates of Q_\theta on a small subset of Dataset D
#
##########################################################################################################
#import create_json_reference

#epochs_Bphi_with_MC = 1000

# I select 1665 images in the val set which saved in ./data: "val_images_captions.pkl",
# to train the B_phi, here is the reference json file path
#refer_1665_save_path = './data/reference_1665.json'

#eval_ids_to_imgNames_save_path = './data/eval_ids_to_imgNames.pkl'

def Sample_Q_with_MC():
    model_path = os.path.join('./models', 'model_MLP-200')

    n_words = len(idx_to_word)

    val_images_names = val_images_captions.keys()

    print "Begin compute Q rewards of {} images...".format(len(val_images_names))

    # create_json_reference.py
    # create_refer(train_images_captions_path, train_images_names, refer_1665_save_path)
    #create_json_reference.create_refer(val_images_captions_path, val_images_names, refer_1665_save_path)

    #with open(eval_ids_to_imgNames_save_path, 'r') as fr_1:
    #    eval_ids_to_imgNames = pickle.load(fr_1)
    #eval_imgNames_to_ids = {}
    #for key, val in eval_ids_to_imgNames.iteritems():
    #    eval_imgNames_to_ids[val] = key

    #with open('./data/train_images_captions_index.pkl', 'r') as fr_2:
    #    train_images_captions_index = pickle.load(fr_2)

    # open the dict that map the image names to image ids
    with open('./data/train_val_imageNames_to_imageIDs.pkl', 'r') as fr:
        train_val_imageNames_to_imageIDs = pickle.load(fr)

    model = CNN_LSTM(n_words = n_words,
                     batch_size = 1,
                     feats_dim = feats_dim,
                     project_dim = project_dim,
                     lstm_size = lstm_size,
                     word_embed_dim = word_embed_dim,
                     lstm_step = lstm_step,
                     bias_init_vector = bias_init_vector)

    tf_images, tf_gen_sentences, tf_all_sentences = model.Monte_Carlo_Rollout()
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    all_images_Q_rewards = {}
    for idx, img_name in enumerate(val_images_names):
        print("current image idx: {},  {}".format(idx, img_name))
        start_time = time.time()

        # Load reference json file
        annFile = './train_val_reference_json/' + img_name + '.json'
        coco = COCO(annFile)

        all_images_Q_rewards[img_name] = {}
        current_image_rewards = all_images_Q_rewards[img_name]
        current_image_rewards['Bleu_4'] = []
        current_image_rewards['Bleu_3'] = []
        current_image_rewards['Bleu_2'] = []
        current_image_rewards['Bleu_1'] = []

        current_feats = np.load(os.path.join(val_feats_path, img_name+'.npy'))
        current_feats = np.reshape(current_feats, [1, feats_dim])

        gen_sents_index, all_sample_sents = sess.run([tf_gen_sentences, tf_all_sentences], feed_dict={tf_images: current_feats})
        gen_sents = []
        for item in gen_sents_index:
            tmp_word = idx_to_word[item]
            tmp_word = tmp_word.replace('\\', '')
            tmp_word = tmp_word.replace('\n', '')
            tmp_word = tmp_word.replace('"', '')
            gen_sents.append(tmp_word)
        gen_sents_list = gen_sents
        punctuation = np.argmax(np.array(gen_sents) == '<eos>') + 1
        gen_sents = gen_sents[:punctuation]
        gen_sents = ' '.join(gen_sents)
        gen_sents = gen_sents.replace(' <eos>', '')
        gen_sents = gen_sents.replace(' ,', ',')
        print "\ngenerated sentences: {}".format(gen_sents)
        
        for i_s, samples in enumerate(all_sample_sents):
            print "\n=========================================================================="
            print "{} / {}".format(i_s, len(all_sample_sents))

            samples = np.asarray(samples)
            sample_sent_1 = []; sample_sent_2 = []; sample_sent_3 = []

            for each_gen_sents_word in gen_sents_list[0: (i_s+1)]:
                sample_sent_1.append(each_gen_sents_word)
                sample_sent_2.append(each_gen_sents_word)
                sample_sent_3.append(each_gen_sents_word)

            for j_s in range(samples.shape[0]):
                word_1, word_2, word_3 = idx_to_word[samples[j_s, 0]], idx_to_word[samples[j_s, 1]], idx_to_word[samples[j_s, 2]]
                word_1, word_2, word_3 = word_1.replace('\n', ''), word_2.replace('\n', ''), word_3.replace('\n', '')
                word_1, word_2, word_3 = word_1.replace('"', ''), word_2.replace('"', ''), word_3.replace('"', '')
                word_1, word_2, word_3 = word_1.replace('\\', ''), word_2.replace('\\', ''), word_3.replace('\\', '')
                sample_sent_1.append(word_1)
                sample_sent_2.append(word_2)
                sample_sent_3.append(word_3)

            sample_sent_1.append('<eos>')
            sample_sent_2.append('<eos>')
            sample_sent_3.append('<eos>')

            three_sample_sents = [sample_sent_1, sample_sent_2, sample_sent_3]

            three_sample_rewards = {}
            three_sample_rewards['Bleu_1'] = 0.0
            three_sample_rewards['Bleu_2'] = 0.0
            three_sample_rewards['Bleu_3'] = 0.0
            three_sample_rewards['Bleu_4'] = 0.0

            for ii, each_sample_sent in enumerate(three_sample_sents):
                if ' ' in each_sample_sent:
                    each_sample_sent.remove(' ') # remove the space element in a list!

                print "sample sentence {},  {}".format(ii, each_sample_sent)

                punctuation = np.argmax(np.array(each_sample_sent) == '<eos>') + 1
                each_sample_sent = each_sample_sent[:punctuation]
                each_sample_sent = ' '.join(each_sample_sent)
                each_sample_sent = each_sample_sent.replace(' <eos>', '')
                each_sample_sent = each_sample_sent.replace(' ,', ',')
                print each_sample_sent
                fw_1 = open("./data/results_MC.json", 'w')
                fw_1.write('[{"image_id": ' + str(train_val_imageNames_to_imageIDs[img_name]) + ', "caption": "' + each_sample_sent + '"}]')
                fw_1.close()

                #annFile = './data/reference_1665.json'
                resFile = './data/results_MC.json'
                #coco = COCO(annFile)
                cocoRes = coco.loadRes(resFile)
                cocoEval = COCOEvalCap(coco, cocoRes)
                cocoEval.params['image_id'] = cocoRes.getImgIds()
                cocoEval.evaluate()

                for metric, score in cocoEval.eval.items():
                    print '%s: %.3f'%(metric, score)
                    if metric == 'Bleu_1':
                        three_sample_rewards['Bleu_1'] += score
                    if metric == 'Bleu_2':
                        three_sample_rewards['Bleu_2'] += score
                    if metric == 'Bleu_3':
                        three_sample_rewards['Bleu_3'] += score
                    if metric == 'Bleu_4':
                        three_sample_rewards['Bleu_4'] += score

            current_image_rewards['Bleu_1'].append(three_sample_rewards['Bleu_1']/3.0)
            current_image_rewards['Bleu_2'].append(three_sample_rewards['Bleu_2']/3.0)
            current_image_rewards['Bleu_3'].append(three_sample_rewards['Bleu_3']/3.0)
            current_image_rewards['Bleu_4'].append(three_sample_rewards['Bleu_4']/3.0)

        # If be in a terminal state, we define Q(g_{1:T}, EOS) = R(g_{1:T})
        fw_1 = open("./data/results_MC.json", 'w')
        fw_1.write('[{"image_id": ' + str(train_val_imageNames_to_imageIDs[img_name]) + ', "caption": "' + gen_sents + '"}]')
        fw_1.close()
        #annFile = './data/reference_1665.json'
        resFile = './data/results_MC.json'
        #coco = COCO(annFile)
        cocoRes = coco.loadRes(resFile)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        cocoEval.evaluate()
        for metric, score in cocoEval.eval.items():
            print '%s: %.3f'%(metric, score)
            if metric == 'Bleu_1':
                current_image_rewards['Bleu_1'].append(score)
            if metric == 'Bleu_2':
                current_image_rewards['Bleu_2'].append(score)
            if metric == 'Bleu_3':
                current_image_rewards['Bleu_3'].append(score)
            if metric == 'Bleu_4':
                current_image_rewards['Bleu_4'].append(score)
        print "Time cost: {}".format(time.time()-start_time)

    with open('./data/all_images_Q_rewards.pkl', 'w') as fw_1:
        pickle.dump(all_images_Q_rewards, fw_1)

def Train_Bphi_Model():
    n_words = len(idx_to_word)

    with open('./data/all_images_Q_rewards.pkl', 'r') as fr_3:
        all_images_Q_rewards = pickle.load(fr_3)

    subset_images_names = all_images_Q_rewards.keys()

    model = CNN_LSTM(n_words = n_words,
                     batch_size = 1,
                     feats_dim = feats_dim,
                     project_dim = project_dim,
                     lstm_size = lstm_size,
                     word_embed_dim = word_embed_dim,
                     lstm_step = lstm_step,
                     bias_init_vector = bias_init_vector)

    Bphi_tf_images, Bphi_tf_Bleu_1, Bphi_tf_Bleu_2, Bphi_tf_Bleu_3, Bphi_tf_Bleu_4, Bphi_tf_loss = model.train_Bphi_model()

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(Bphi_tf_loss)
    sess = tf.InteractiveSession()
    #tf.initialize_all_variables().run()
    new_saver = tf.train.Saver(max_to_keep=500)
    #new_saver = tf.train.import_meta_graph('./models/model-32.meta')
    #new_saver.restore(sess, tf.train.latest_checkpoint('./models'))
    new_saver.restore(sess, './models/model-50')

    loss_to_draw = []
    for epoch in range(0, epochs_Bphi_with_MC):
        loss_to_draw_epoch = []
        random.shuffle(subset_images_names)

        for start, end in zip(range(0, len(subset_images_names), 1),
                              range(1, len(subset_images_names), 1)):
            start_time_batch = time.time()

            current_feats = []

            # Bleu_1, Bleu_2, Bleu_3, Bleu_4
            current_Bleu_1 = []
            current_Bleu_2 = []
            current_Bleu_3 = []
            current_Bleu_4 = []

            img_names = subset_images_names[start:end]
            for each_img_name in img_names:
                img_feat = np.load(os.path.join(train_val_feats_path, each_img_name+'.npy'))
                current_feats.append(img_feat)

                current_Bleu_1.append(all_images_Q_rewards[each_img_name]['Bleu_1'])
                current_Bleu_2.append(all_images_Q_rewards[each_img_name]['Bleu_2'])
                current_Bleu_3.append(all_images_Q_rewards[each_img_name]['Bleu_3'])
                current_Bleu_4.append(all_images_Q_rewards[each_img_name]['Bleu_4'])

            current_feats = np.asarray(current_feats, dtype=np.float32)
            current_Bleu_1 = np.asarray(current_Bleu_1, dtype=np.float32)
            current_Bleu_2 = np.asarray(current_Bleu_2, dtype=np.float32)
            current_Bleu_3 = np.asarray(current_Bleu_3, dtype=np.float32)
            current_Bleu_4 = np.asarray(current_Bleu_4, dtype=np.float32)

            _, loss_val = sess.run([train_op, Bphi_tf_loss],
                                   feed_dict = {Bphi_tf_images: current_feats,
                                                Bphi_tf_Bleu_1: current_Bleu_1,
                                                Bphi_tf_Bleu_2: current_Bleu_2,
                                                Bphi_tf_Bleu_3: current_Bleu_3,
                                                Bphi_tf_Bleu_4: current_Bleu_4
                                       })

            loss_to_draw_epoch.append(loss_val[0,0])
            print "idx: {}  epoch: {}  loss: {}  Time cost: {}".format(start, epoch, loss_val[0,0], time.time() - start_time_batch)

        loss_to_draw.append(np.mean(loss_to_draw_epoch))
        plt_save_img_name = 'Bphi_train_' + str(epoch) + '.png'
        plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
        plt.grid(True)
        plt.savefig(os.path.join('./loss_imgs', plt_save_img_name))

        if np.mod(epoch, 2) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            new_saver.save(sess, os.path.join('./models', 'Bphi_train_model'), global_step=epoch)


##############################################################################################################
#
# Step 4:  go through all the images in D, SGD update of \theta, \phi
#
##############################################################################################################
def Train_SGD_update():
    model_path = os.path.join('./models', 'Bphi_train_model-84')
    batch_num_images = 100 # 100
    epoches = n_epochs # 500
    n_words = len(idx_to_word)
    train_images_names = train_images_captions.keys()

    # open the dict that map the image names to image ids
    with open('./data/train_val_imageNames_to_imageIDs.pkl', 'r') as fr:
        train_val_imageNames_to_imageIDs = pickle.load(fr)

    # Load COCO reference json file
    annFile = './data/train_val_all_reference.json'
    coco = COCO(annFile)

    # model initialization
    model = CNN_LSTM(n_words = n_words,
                     batch_size = batch_num_images,
                     feats_dim = feats_dim,
                     project_dim = project_dim,
                     lstm_size = lstm_size,
                     word_embed_dim = word_embed_dim,
                     lstm_step = lstm_step,
                     bias_init_vector = bias_init_vector)

    # The first model is used to generate sample sentences and Baselines.
    # Then we use the sample sentences and coco caption API to compute the Q_rewards.
    # And the second model is used to transfer the Q_rewards, Baselines values,
    # the loss function is \sum(log(max_probability) * rewards)
    tf_images, tf_gen_sents_index, tf_all_sample_sents, tf_all_baselines = model.Monte_Carlo_and_Baseline()
    tf_images_2, tf_Q_rewards, tf_Baselines, tf_loss, tf_max_prob, tf_current_rewards, tf_logit_words = model.SGD_update(batch_num_images=1000)

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    #tf.initialize_all_variables().run()

    # save every epoch loss value in loss_to_draw
    loss_to_draw = []
    for epoch in range(0, epoches):
        # save every batch loss value in loss_to_draw_epoch
        loss_to_draw_epoch = []

        # shuffle the order of images randomly
        random.shuffle(train_images_names)

        # store rewards of all the training images
        train_val_images_Q_rewards = {}

        for start, end in zip(range(0, len(train_images_names), batch_num_images),
                              range(batch_num_images, len(train_images_names), batch_num_images)):
            start_time = time.time()

            img_names = train_images_names[start:end]
            current_feats = []
            for img_name in img_names:
                tmp_feats = np.load(os.path.join(train_val_feats_path, img_name+'.npy'))
                current_feats.append(tmp_feats)
            current_feats = np.asarray(current_feats)

            # store rewards of all the training images
            #train_val_images_Q_rewards = {}
            #ONE IMAGE: for idx, img_name in enumerate(train_images_names):
            #ONE IMAGE: print "{},  {}".format(idx, img_name)
            #ONE IMAGE: start_time = time.time()
            current_batch_rewards = {}
            current_batch_rewards['Bleu_1'] = []
            current_batch_rewards['Bleu_2'] = []
            current_batch_rewards['Bleu_3'] = []
            current_batch_rewards['Bleu_4'] = []

            # weighted sum
            sum_image_rewards = []
            Bleu_1_weight = 0.5
            Bleu_2_weight = 0.5
            Bleu_3_weight = 1.0
            Bleu_4_weight = 1.0

            #ONE IMAGE: current_feats = np.load(os.path.join(train_val_feats_path, img_name+'.npy'))
            #ONE IMAGE: current_feats = np.reshape(current_feats, [1, feats_dim])
            
            
            ###################################################################################################################################
            # 
            # Below, for the current 100 images, we compute Q(g1:t-1, gt) for gt with K Monte Carlo rollouts, using Equation (6)
            # Meanwhile, we compute estimated baseline B_phi(g1:t-1)
            #
            ###################################################################################################################################
            feed_dict = {tf_images: current_feats}
            gen_sents_index, all_sample_sents, all_baselines = sess.run([tf_gen_sents_index, tf_all_sample_sents, tf_all_baselines], feed_dict)
            
            # 100 sentences, every sentence has 30 words, thus its shape is 100 x 30
            batch_sentences = []
            for tmp_i in range(0, batch_num_images):
                single_sentences = []
                for tmp_j in range(0, len(gen_sents_index)):
                    word_idx = gen_sents_index[tmp_j][tmp_i]
                    word = idx_to_word[word_idx]
                    word = word.replace('\n', '')
                    word = word.replace('\\', '')
                    word = word.replace('"', '')
                    single_sentences.append(word)
                batch_sentences.append(single_sentences)

            #ONE IMAGE: tmp_sentences = map(lambda x: idx_to_word[x], gen_sents_index)
            #ONE IMAGE: print tmp_sentences
            #ONE IMAGE: sentences = []
            #ONE IMAGE: for word in tmp_sentences:
            #ONE IMAGE:     word = word.replace('\n', '')
            #ONE IMAGE:     word = word.replace('\\', '')
            #ONE IMAGE:     word = word.replace('"', '')
            #ONE IMAGE:     sentences.append(word)
            
            batch_sentences_processed = []
            #gen_sents_list = batch_sentences
            for tmp_i in range(0, batch_num_images):
                tmp_sentences = batch_sentences[tmp_i]
                punctuation = np.argmax(np.array(tmp_sentences) == '<eos>') + 1
                tmp_sentences = tmp_sentences[:punctuation]
                tmp_sentences = ' '.join(tmp_sentences)
                tmp_sentences = tmp_sentences.replace('<bos> ', '')
                tmp_sentences = tmp_sentences.replace(' <eos>', '')
                batch_sentences_processed.append(tmp_sentences)
                #print "Idx: {}  Image Name: {}  Gen Sentence: {}".format(tmp_i, img_names[tmp_i], generated_sentence)
            
            #ONE IMAGE: gen_sents_list = sentences
            #ONE IMAGE: punctuation = np.argmax(np.array(sentences) == '<eos>') + 1
            #ONE IMAGE: sentences = sentences[:punctuation]
            #ONE IMAGE: generated_sentence = ' '.join(sentences)
            #ONE IMAGE: generated_sentence = generated_sentence.replace('<bos> ', '')
            #ONE IMAGE: generated_sentence = generated_sentence.replace(' <eos>', '')
            #ONE IMAGE: print "Generated sentences: {}".format(generated_sentence)
            
            # 0, 1, 2, ..., 28, the 30th is computed by the whole generated sentences
            for time_step in range(0, lstm_step-1):
                print "\n===================================================================================================="
                print "Time step:  {} \n".format(time_step)
                batch_samples = all_sample_sents[time_step]
                batch_samples = np.asarray(batch_samples)

                batch_sample_sents_1 = []
                batch_sample_sents_2 = []
                batch_sample_sents_3 = []
                # store the sample sentences, each sample list has 100 images' sentences
                for img_idx in range(0, batch_num_images):
                    batch_sample_sents_1.append([])
                    batch_sample_sents_2.append([])
                    batch_sample_sents_3.append([])

                # 0, 1, 2, ..., 99
                for img_idx in range(0, batch_num_images):
                    for each_gen_sents_word in batch_sentences[img_idx][0:time_step+1]:
                        each_gen_sents_word = each_gen_sents_word.replace('\n', '')
                        each_gen_sents_word = each_gen_sents_word.replace('\\', '')
                        each_gen_sents_word = each_gen_sents_word.replace('"', '')
                        batch_sample_sents_1[img_idx].append(each_gen_sents_word)
                        batch_sample_sents_2[img_idx].append(each_gen_sents_word)
                        batch_sample_sents_3[img_idx].append(each_gen_sents_word)

                # 0, 1, 2, ..., 99
                for img_idx in range(0, batch_num_images):
                    for tmp_i in range(0, batch_samples.shape[0]):
                        word_1 = idx_to_word[batch_samples[tmp_i, img_idx, 0]]
                        word_2 = idx_to_word[batch_samples[tmp_i, img_idx, 1]]
                        word_3 = idx_to_word[batch_samples[tmp_i, img_idx, 2]]
                        word_1, word_2, word_3 = word_1.replace('\n', ''), word_2.replace('\n', ''), word_3.replace('\n', '')
                        word_1, word_2, word_3 = word_1.replace('\\', ''), word_2.replace('\\', ''), word_3.replace('\\', '')
                        word_1, word_2, word_3 = word_1.replace('"', ''), word_2.replace('"', ''), word_3.replace('"', '')

                        batch_sample_sents_1[img_idx].append(word_1)
                        batch_sample_sents_2[img_idx].append(word_2)
                        batch_sample_sents_3[img_idx].append(word_3)
                    batch_sample_sents_1[img_idx].append('<eos>')
                    batch_sample_sents_2[img_idx].append('<eos>')
                    batch_sample_sents_3[img_idx].append('<eos>')

                batch_three_sample_sents = [batch_sample_sents_1, batch_sample_sents_2, batch_sample_sents_3]
                three_sample_rewards = {}
                three_sample_rewards['Bleu_1'] = 0.0
                three_sample_rewards['Bleu_2'] = 0.0
                three_sample_rewards['Bleu_3'] = 0.0
                three_sample_rewards['Bleu_4'] = 0.0
                
                for tmp_i, batch_sample_sents in enumerate(batch_three_sample_sents):
                    ######################################################################################
                    # write the sample sentences of current 100 images
                    ######################################################################################
                    fw_1 = open("./data/results_batch_sample_sents.json", 'w')
                    fw_1.write('[')
                    
                    for img_idx in range(0, batch_num_images):
                        if ' ' in batch_sample_sents[img_idx]:
                            batch_sample_sents[img_idx].remove(' ')
                        
                        punctuation = np.argmax(np.array(batch_sample_sents[img_idx]) == '<eos>') + 1
                        batch_sample_sents[img_idx] = batch_sample_sents[img_idx][:punctuation]
                        batch_sample_sents[img_idx] = ' '.join(batch_sample_sents[img_idx])
                        batch_sample_sents[img_idx] = batch_sample_sents[img_idx].replace(' <eos>', '')
                        batch_sample_sents[img_idx] = batch_sample_sents[img_idx].replace(' ,', ',')
                        
                        if img_idx != batch_num_images-1:
                            fw_1.write('{"image_id": ' + str(train_val_imageNames_to_imageIDs[img_names[img_idx]]) + ', "caption": "' + batch_sample_sents[img_idx] + '"}, ')
                        else:
                            fw_1.write('{"image_id": ' + str(train_val_imageNames_to_imageIDs[img_names[img_idx]]) + ', "caption": "' + batch_sample_sents[img_idx] + '"}]')
                    fw_1.close()
                    
                    ########################################################################################
                    # compute the Bleu1,2,3,4 score using current 100 images
                    ########################################################################################
                    #annFile = './data/train_val_all_reference.json'
                    resFile = './data/results_batch_sample_sents.json'
                    #coco = COCO(annFile)
                    cocoRes = coco.loadRes(resFile)
                    cocoEval = COCOEvalCap(coco, cocoRes)
                    cocoEval.params['image_id'] = cocoRes.getImgIds()
                    cocoEval.evaluate()
                    for metric, score in cocoEval.eval.items():
                        if metric == 'Bleu_1':
                            three_sample_rewards['Bleu_1'] += score
                        if metric == 'Bleu_2':
                            three_sample_rewards['Bleu_2'] += score
                        if metric == 'Bleu_3':
                            three_sample_rewards['Bleu_3'] += score
                        if metric == 'Bleu_4':
                            three_sample_rewards['Bleu_4'] += score
            
                current_batch_rewards['Bleu_1'].append(three_sample_rewards['Bleu_1']/3.0)
                current_batch_rewards['Bleu_2'].append(three_sample_rewards['Bleu_2']/3.0)
                current_batch_rewards['Bleu_3'].append(three_sample_rewards['Bleu_3']/3.0)
                current_batch_rewards['Bleu_4'].append(three_sample_rewards['Bleu_4']/3.0)

            #####################################################################################################
            # compute the 30th rewards of the current 100 images
            #####################################################################################################
            fw_2 = open("./data/results_batch_sample_sents.json", 'w')
            fw_2.write('[')
            for img_idx in range(0, batch_num_images):
                if img_idx != batch_num_images-1:
                    fw_2.write('{"image_id": ' + str(train_val_imageNames_to_imageIDs[img_names[img_idx]]) + ', "caption": "' + batch_sentences_processed[img_idx] + '"}, ')
                else:
                    fw_2.write('{"image_id": ' + str(train_val_imageNames_to_imageIDs[img_names[img_idx]]) + ', "caption": "' + batch_sentences_processed[img_idx] + '"}]')
            fw_2.close()
            #annFile = './data/train_val_all_reference.json'
            resFile = './data/results_batch_sample_sents.json'
            #coco = COCO(annFile)
            cocoRes = coco.loadRes(resFile)
            cocoEval = COCOEvalCap(coco, cocoRes)
            cocoEval.params['image_id'] = cocoRes.getImgIds()
            cocoEval.evaluate()
            for metric, score in cocoEval.eval.items():
                if metric == 'Bleu_1':
                    current_batch_rewards['Bleu_1'].append(score)
                if metric == 'Bleu_2':
                    current_batch_rewards['Bleu_2'].append(score)
                if metric == 'Bleu_3':
                    current_batch_rewards['Bleu_3'].append(score)
                if metric == 'Bleu_4':
                    current_batch_rewards['Bleu_4'].append(score)
           
            # compute the weight sum of Bleu value as rewards
            for tmp_idx in range(0, lstm_step):
                tmp_reward = current_batch_rewards['Bleu_1'][tmp_idx] * Bleu_1_weight + \
                             current_batch_rewards['Bleu_2'][tmp_idx] * Bleu_2_weight + \
                             current_batch_rewards['Bleu_3'][tmp_idx] * Bleu_3_weight + \
                             current_batch_rewards['Bleu_4'][tmp_idx] * Bleu_4_weight
                sum_image_rewards.append(tmp_reward)
            sum_image_rewards = np.asarray(sum_image_rewards)
            #sum_image_rewards = np.reshape(sum_image_rewards, [1, lstm_step])
            sum_image_rewards = np.array([sum_image_rewards, ] * batch_num_images)

            all_baselines = np.asarray(all_baselines)
            all_baselines = np.reshape(all_baselines, [batch_num_images, lstm_step])
            #all_baselines_mean = np.mean(all_baselines, axis=0)
            #all_baselines = np.array([all_baselines_mean,] * batch_num_images)
            feed_dict = {tf_images_2: current_feats, tf_Q_rewards: sum_image_rewards, tf_Baselines: all_baselines}
            _, loss_value, max_prob, current_rewards, logit_words = sess.run([train_op, tf_loss, tf_max_prob, tf_current_rewards, tf_logit_words], feed_dict)
            #ipdb.set_trace()
            loss_to_draw_epoch.append(loss_value)
            print "idx: {}  epoch: {}  loss: {}  Time cost: {}".format(start, epoch, loss_value, time.time()-start_time)

        # draw loss curve every epoch
        loss_to_draw.append(np.mean(loss_to_draw_epoch))
        plt_save_img_name = 'SGD_update_' + str(epoch) + '.png'
        plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
        plt.grid(True)
        plt.savefig(os.path.join(loss_images_save_path, plt_save_img_name))

        if np.mod(epoch, 1) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            saver.save(sess, os.path.join('./models', 'SGD_update_model'), global_step=epoch)

            #ONE IMAGE: # compute the 29 rewards using all_sample_sents
            #ONE IMAGE: # the 30th reward is computed with gen_sents_list
            #ONE IMAGE: for t in range(0, lstm_step-1):
            #ONE IMAGE:     samples = all_sample_sents[t]
            #ONE IMAGE:     samples = np.asarray(samples)

            #ONE IMAGE:     sample_sent_1 = []
            #ONE IMAGE:     sample_sent_2 = []
            #ONE IMAGE:     sample_sent_3 = []
            #ONE IMAGE:     for each_gen_sents_word in gen_sents_list[0:t+1]:
            #ONE IMAGE:         sample_sent_1.append(each_gen_sents_word)
            #ONE IMAGE:         sample_sent_2.append(each_gen_sents_word)
            #ONE IMAGE:         sample_sent_3.append(each_gen_sents_word)

            #ONE IMAGE:     for i in range(samples.shape[0]):
            #ONE IMAGE:         word_1, word_2, word_3 = idx_to_word[samples[i, 0]], idx_to_word[samples[i, 1]], idx_to_word[samples[i, 2]]

            #ONE IMAGE:         word_1, word_2, word_3 = word_1.replace('\n', ''), word_2.replace('\n', ''), word_3.replace('\n', '')
            #ONE IMAGE:         word_1, word_2, word_3 = word_1.replace('\\', ''), word_2.replace('\\', ''), word_3.replace('\\', '')
            #ONE IMAGE:         word_1, word_2, word_3 = word_1.replace('"', ''), word_2.replace('"', ''), word_3.replace('"', '')

            #ONE IMAGE:         sample_sent_1.append(word_1)
            #ONE IMAGE:         sample_sent_2.append(word_2)
            #ONE IMAGE:         sample_sent_3.append(word_3)

            #ONE IMAGE:     sample_sent_1.append('<eos>')
            #ONE IMAGE:     sample_sent_2.append('<eos>')
            #ONE IMAGE:     sample_sent_3.append('<eos>')

            #ONE IMAGE:     three_sample_sents = [sample_sent_1, sample_sent_2, sample_sent_3]
            #ONE IMAGE:     three_sample_rewards = {}
            #ONE IMAGE:     three_sample_rewards['Bleu_1'] = 0.0
            #ONE IMAGE:     three_sample_rewards['Bleu_2'] = 0.0
            #ONE IMAGE:     three_sample_rewards['Bleu_3'] = 0.0
            #ONE IMAGE:     three_sample_rewards['Bleu_4'] = 0.0

            #ONE IMAGE:     for i, each_sample_sent in enumerate(three_sample_sents):
            #ONE IMAGE:         # remove the space element in a list
            #ONE IMAGE:         if ' ' in each_sample_sent:
            #ONE IMAGE:             each_sample_sent.remove(' ')

            #ONE IMAGE:         punctuation = np.argmax(np.array(each_sample_sent) == '<eos>') + 1
            #ONE IMAGE:         each_sample_sent = each_sample_sent[:punctuation]
            #ONE IMAGE:         each_sample_sent = ' '.join(each_sample_sent)
            #ONE IMAGE:         each_sample_sent = each_sample_sent.replace(' <eos>', '')
            #ONE IMAGE:         each_sample_sent = each_sample_sent.replace(' ,', ',')

            #ONE IMAGE:         fw_1 = open("./data/results_each_sample_sent.json", 'w')
            #ONE IMAGE:         fw_1.write('[{"image_id": ' + str(train_val_imageNames_to_imageIDs[img_name]) + ', "caption": "' + each_sample_sent + '"}]')
            #ONE IMAGE:         fw_1.close()

            #ONE IMAGE:         annFile = './train_val_reference_json/' + img_name + '.json'
            #ONE IMAGE:         resFile = './data/results_each_sample_sent.json'
            #ONE IMAGE:         coco = COCO(annFile)
            #ONE IMAGE:         cocoRes = coco.loadRes(resFile)
            #ONE IMAGE:         cocoEval = COCOEvalCap(coco, cocoRes)
            #ONE IMAGE:         cocoEval.params['image_id'] = cocoRes.getImgIds()
            #ONE IMAGE:         cocoEval.evaluate()
            #ONE IMAGE:         for metric, score in cocoEval.eval.items():
            #ONE IMAGE:             if metric == 'Bleu_1':
            #ONE IMAGE:                 three_sample_rewards['Bleu_1'] += score
            #ONE IMAGE:             if metric == 'Bleu_2':
            #ONE IMAGE:                 three_sample_rewards['Bleu_2'] += score
            #ONE IMAGE:             if metric == 'Bleu_3':
            #ONE IMAGE:                 three_sample_rewards['Bleu_3'] += score
            #ONE IMAGE:             if metric == 'Bleu_4':
            #ONE IMAGE:                 three_sample_rewards['Bleu_4'] += score

            #ONE IMAGE:     current_image_rewards['Bleu_1'].append(three_sample_rewards['Bleu_1']/3.0)
            #ONE IMAGE:     current_image_rewards['Bleu_2'].append(three_sample_rewards['Bleu_2']/3.0)
            #ONE IMAGE:     current_image_rewards['Bleu_3'].append(three_sample_rewards['Bleu_3']/3.0)
            #ONE IMAGE:     current_image_rewards['Bleu_4'].append(three_sample_rewards['Bleu_4']/3.0)

            #ONE IMAGE: fw_1 = open("./data/results_each_sample_sent.json", 'w')
            #ONE IMAGE: fw_1.write('[{"image_id": ' + str(train_val_imageNames_to_imageIDs[img_name]) + ', "caption": "' + generated_sentence + '"}]')
            #ONE IMAGE: fw_1.close()

            #ONE IMAGE: annFile = './train_val_reference_json/' + img_name + '.json'
            #ONE IMAGE: resFile = './data/results_each_sample_sent.json'
            #ONE IMAGE: coco = COCO(annFile)
            #ONE IMAGE: cocoRes = coco.loadRes(resFile)
            #ONE IMAGE: cocoEval = COCOEvalCap(coco, cocoRes)
            #ONE IMAGE: cocoEval.params['image_id'] = cocoRes.getImgIds()
            #ONE IMAGE: cocoEval.evaluate()
            #ONE IMAGE: for metric, score in cocoEval.eval.items():
            #ONE IMAGE:     if metric == 'Bleu_1':
            #ONE IMAGE:         current_image_rewards['Bleu_1'].append(score)
            #ONE IMAGE:     if metric == 'Bleu_2':
            #ONE IMAGE:         current_image_rewards['Bleu_2'].append(score)
            #ONE IMAGE:     if metric == 'Bleu_3':
            #ONE IMAGE:         current_image_rewards['Bleu_3'].append(score)
            #ONE IMAGE:     if metric == 'Bleu_4':
            #ONE IMAGE:         current_image_rewards['Bleu_4'].append(score)

            #ONE IMAGE: # save the rewards immediately
            #ONE IMAGE: train_val_images_Q_rewards[img_name] = current_image_rewards
            #ONE IMAGE: with open('./data/train_val_images_Q_rewards.pkl', 'w') as fw_2:
            #ONE IMAGE:     pickle.dump(train_val_images_Q_rewards, fw_2)

            #ONE IMAGE: # compute the weight sum of Bleu value as rewards
            #ONE IMAGE: for tmp_idx in range(0, lstm_step):
            #ONE IMAGE:     tmp_reward = current_image_rewards['Bleu_1'][tmp_idx] * Bleu_1_weight + \
            #ONE IMAGE:                  current_image_rewards['Bleu_2'][tmp_idx] * Bleu_2_weight + \
            #ONE IMAGE:                  current_image_rewards['Bleu_3'][tmp_idx] * Bleu_3_weight + \
            #ONE IMAGE:                  current_image_rewards['Bleu_4'][tmp_idx] * Bleu_4_weight
            #ONE IMAGE:     sum_image_rewards.append(tmp_reward)

            #ONE IMAGE: sum_image_rewards = np.asarray(sum_image_rewards)
            #ONE IMAGE: sum_image_rewards = np.reshape(sum_image_rewards, [1, lstm_step])
            #ONE IMAGE: all_baselines = np.asarray(all_baselines)
            #ONE IMAGE: all_baselines = np.reshape(all_baselines, [1, lstm_step])
            #ONE IMAGE: feed_dict = {tf_images_2: current_feats, tf_Q_rewards: sum_image_rewards, tf_Baselines: all_baselines}
            #ONE IMAGE: _, loss_value = sess.run([train_op, tf_loss], feed_dict)

            #ONE IMAGE: loss_to_draw_epoch.append(loss_value)

            #ONE IMAGE: print "idx: {}  epoch: {}  loss: {}  Time cost: {}".format(idx, epoch, loss_value, time.time()-start_time)

        #ONE IMAGE: # draw loss curve every epoch
        #ONE IMAGE: loss_to_draw.append(np.mean(loss_to_draw_epoch))
        #ONE IMAGE: plt_save_img_name = str(epoch) + '.png'
        #ONE IMAGE: plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
        #ONE IMAGE: plt.grid(True)
        #ONE IMAGE: plt.savefig(os.path.join(loss_images_save_path, plt_save_img_name))

        #ONE IMAGE: if np.mod(epoch, 2) == 0:
        #ONE IMAGE:     print "Epoch ", epoch, " is done. Saving the model ..."
        #ONE IMAGE:     saver.save(sess, os.path.join('./models', 'SGD_update_model'), global_step=epoch)



