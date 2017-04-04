# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 19:35:28 2017

@author: mrins
"""

#! /usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from rnn_language_modeling_theano import RNNTheano
from rnn import vanillaRNNNet1
from lstm import LSTMNet1, LSTMNet2, StackedLSTMNet1
from gru import GRUNet1

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')



def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print(time ,": Loss after num_examples_seen=", num_examples_seen, "epoch=", epoch,": ", loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print("Setting learning rate to ", learning_rate)
            sys.stdout.flush()
            # ADDED! Saving model oarameters
#            save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"


print("Reading CSV file...")
with open("Data/sample_reddit_comments.csv") as f:
    reader = csv.reader(f, skipinitialspace = True)
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    sentences = ["%s %s %s" %(sentence_start_token, x, sentence_end_token) for x in sentences]
print("Parsed ", (len(sentences)), " sentences")

tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found ",len(word_freq.items()), " unique words tokens.")

vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print("Using vocabulary size ", vocabulary_size)
print("The least frequent word in our vocabulary is", vocab[-1][0], "and appeared", vocab[-1][1]," times.")

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print("Example sentence: ", sentences[0])
print("Example sentence after Pre-processing: ",tokenized_sentences[0])


# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])





#model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
#model = vanillaRNNNet1(n_in=vocabulary_size, n_hidden=_HIDDEN_DIM, n_out=vocabulary_size)
model = StackedLSTMNet1(n_in=vocabulary_size, n_hidden=_HIDDEN_DIM, n_out=vocabulary_size)
#model = GRUNet1(n_in=vocabulary_size, n_hidden=_HIDDEN_DIM, n_out=vocabulary_size)
t1 = time.time()
model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
t2 = time.time()
print("SGD Step time:", (t2 - t1) * 1000.," milliseconds")


train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)