# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 19:56:59 2017

@author: mrins
"""

""" Regressor """

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from rnn_v1 import vanillaRNNNet
from Train import train


datatype = theano.config.floatX


def test_vanillaRNN_on_continuous_data(data, model_params):


    ###########################
    ####### DECLARATIONS  #####
    ###########################    
    
    index = T.scalar('index')
    X = T.matrix('X')
    y = T.matrix('y')
    lr = 0.01
    n_epochs = 4
    n_in, n_hidden, n_out = model_params
    
    ###########################
    ####### LOAD DATASETS #####
    ###########################
    dataset = split_load_dataset(data)
    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
#    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
#    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        
    validation_frequency = n_train_batches
    ###########################
    # BUILD TRAINING MODELS #
    ###########################
        
    print('... Start designing the model')
    
    np.random.seed(0)
    rng = np.random.RandomState(7341)
    
    rnn_1_hidden_layer = vanillaRNNNet(
                                    rng = rng,
                                    X = X,
                                    y = y,
                                    n_in = n_in, # Dimension of each input,
                                    n_hidden = n_hidden,
                                    n_out = n_out, # Dimension of output
                                    lr = lr
                    )
                    
    model = rnn_1_hidden_layer
    print('... Model Design Complete')


    compute_train_error = theano.function(
        inputs = [index],
        outputs = model.cost,
        givens = {
                    X : train_set_x[index], # Dimension (n_steps, n_in)
                    y : train_set_y[index]  # Dimension (n_steps, n_out)
                }
        )
                  
    
#    train_model = theano.function(
#        inputs = [index],
#        outputs = model.cost,
#        updates = model.updates,
#        givens = {
#                    X : train_set_x[index],
#                    y : train_set_y[index]
#                }
#        )
#        
#    test_model = theano.function(
#        inputs = [index],
#        outputs = model.errors(y),
#        givens = {
#                    X : train_set_x[index*batch_sz:(index+1)*batch_sz],
#                    y : train_set_y[index*batch_sz:(index+1)*batch_sz]
#                }        
#        )
#    
#    validate_model = theano.function(
#        inputs = [index],
#        outputs = model.errors(y),
#        givens = {
#                    X : train_set_x[index*batch_sz:(index+1)*batch_sz],
#                    y : train_set_y[index*batch_sz:(index+1)*batch_sz]
#                }        
#        )
#
#
#    ###########################
#    #######   TRAIN/TEST  #####
#    ###########################
#
#    print('... Fitting the model')
#    
#    n_batches = [n_train_batches, n_valid_batches, n_test_batches]
#    train(n_batches, train_fn, validate_fn, test_fn,
#          numEpochs = n_epochs,
#          validationFrequency = validation_frequency)
          
          
          

def split_load_dataset(data_xy):
    
    X, y = data_xy[0], data_xy[1]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
    test_x, valid_x, test_y, valid_y = train_test_split(test_x, test_y, test_size=0.5)
    
    def shared_dataset(data_x, data_y):
        shared_x = theano.shared(np.asarray(data_x, dtype = datatype))
        shared_y = theano.shared(np.asarray(data_y, dtype = datatype))
        return shared_x, shared_y
        
    train_set_x, train_set_y = shared_dataset(train_x, train_y)
    valid_set_x, valid_set_y = shared_dataset(valid_x, valid_y)
    test_set_x, test_set_y = shared_dataset(test_x, test_y)
    
    print(train_set_x.shape, train_x.shape)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval
    
    


          
          
def generate_data():
    """ Test RNN with real-valued outputs. """
    n_hidden = 10
    n_in = 5
    n_out = 3
    n_steps = 10
    n_seq = 100

    np.random.seed(0)
    # simple lag test
    seq = np.random.randn(n_seq, n_steps, n_in)
    targets = np.zeros((n_seq, n_steps, n_out))

    targets[:, 1:, 0] = seq[:, :-1, 3]  # delayed 1
    targets[:, 1:, 1] = seq[:, :-1, 2]  # delayed 1
    targets[:, 2:, 2] = seq[:, :-2, 0]  # delayed 2

    targets += 0.01 * np.random.standard_normal(targets.shape)

    test_vanillaRNN_on_continuous_data([seq, targets], [n_in, n_hidden, n_out])
    
#    model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
#                    learning_rate=0.001, learning_rate_decay=0.999,
#                    n_epochs=400, activation='tanh')
#
#    model.fit(seq, targets, validation_frequency=1000)
#
#    plt.close('all')
#    fig = plt.figure()
#    ax1 = plt.subplot(211)
#    plt.plot(seq[0])
#    ax1.set_title('input')
#
#    ax2 = plt.subplot(212)
#    true_targets = plt.plot(targets[0])
#
#    guess = model.predict(seq[0])
#    guessed_targets = plt.plot(guess, linestyle='--')
#    for i, x in enumerate(guessed_targets):
#        x.set_color(true_targets[i].get_color())
#    ax2.set_title('solid: true output, dashed: model output')
    
    
generate_data()