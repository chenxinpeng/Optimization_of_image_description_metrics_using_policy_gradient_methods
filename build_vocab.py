# encoding: UTF-8

#-----------------------------------------------------------------------
# We preprocess the text data by lower casing, and replacing words which
# occur less than 5 times in the 82K training set with <unk>;
# This results in a vocabulary size of 10,622 (from 32,807 words).
#-----------------------------------------------------------------------

import os
import numpy as np
import cPickle as pickle
import time


train_images_captions_path = './data/train_images_captions.pkl'
with open(train_images_captions_path, 'r') as train_fr:
    train_images_captions = pickle.load(train_fr)

val_images_captions_path = './data/val_images_captions.pkl'
with open(val_images_captions_path, 'r') as val_fr:
    val_images_captions = pickle.load(val_fr)


#------------------------------------------------------------------------
# Borrowed this function from NeuralTalk:
# https://github.com/karpathy/neuraltalk/blob/master/driver.py#L16
#-----------------------------------------------------------------------
def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    print 'Preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )

    t0 = time.time()
    word_counts = {}
    nsents = 0

    for sent in sentence_iterator:
        nsents += 1
        tmp_sent = sent.split(' ')
        # remove the empty string '' in the sentence
        tmp_sent = filter(None, tmp_sent)
        for w in tmp_sent:
           word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'Filter words from %d to %d in %0.2fs' % (len(word_counts), len(vocab), time.time()-t0)

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx + 4
        ixtoword[idx+4] = w

    word_counts['<eos>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<pad>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector


# extract all sentences in captions
all_sents = []
for image, sents in train_images_captions.iteritems():
    for each_sent in sents:
        all_sents.append(each_sent)
#for image, sents in val_images_captions.iteritems():
#    for each_sent in sents:
#        all_sents.append(each_sent)

word_to_idx, idx_to_word, bias_init_vector = preProBuildWordVocab(all_sents, word_count_threshold=5)

with open('./data/idx_to_word.pkl', 'w') as fw_1:
    pickle.dump(idx_to_word, fw_1)

with open('./data/word_to_idx.pkl', 'w') as fw_2:
    pickle.dump(word_to_idx, fw_2)

np.save('./data/bias_init_vector.npy', bias_init_vector)

