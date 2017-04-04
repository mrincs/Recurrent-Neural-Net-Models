# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:16:13 2017

@author: mrins
"""

import numpy as np
import theano
import theano.tensor as T

class SoftMaxLayer(object):

    def __init__(self, input, n_in, n_out, W=None, b=None):

        self.W, self.b = self.initializeParams(W, b, n_in, n_out)
         
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)
        
        self.params = [self.W, self.b]
        
        self.input = input
        
        
    def initializeParams(self, W, b, n_in, n_out):
        
        if W is None:
            W = theano.shared(
                value = np.zeros((n_in, n_out),
                                 dtype = theano.config.floatX),
                        name = "W",
                        borrow = True
                    )            
        if b is None:
            b = theano.shared(
                value = np.zeros((n_out,),
                                 dtype = theano.config.floatX),
                        name = "b",
                        borrow = True
                    )        
        return [W, b]
        
        
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        
        
    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()