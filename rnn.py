# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:22:51 2017

@author: mrins
"""


import numpy as np
import theano
import theano.tensor as T
from LogisticRegression import SoftMaxLayer
from GradUpdates import gradientUpdates


datatype = theano.config.floatX


class vanillaRNNLayer(object):
    
    def __init__(self, rng, input, shape):
        
        self.X = input
        self.n_in, self.n_out = shape
        
        self.W_xg, self.W_hg, self.W_go, self.b_g, self.b_o, self.h0 = self.initializeParams(self.n_in, self.n_out)
        
        self.layerParams = [self.W_xg, self.W_hg, self.W_go, self.b_g, self.b_o]
        
        def forwardProp(x_t, h_tprev): 
            h_t = T.tanh(theano.dot(self.W_xg.T, x_t) + theano.dot(self.W_hg.T, h_tprev) + self.b_g)
            y_t = T.nnet.sigmoid(theano.dot(self.W_go.T, h_t) + self.b_o)
            return h_t, y_t
                               
        self.updates = {}
        for param in self.layerParams:
            self.updates[param] = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=datatype))
            
            
#        self.h, _ = theano.scan(fn = forwardProp,
#                                        sequences = [self.X],
#                                        outputs_info = None,
#                                        non_sequences = self.layerParams
#                                        #strict = True
#                                        )
            
        [self.h, self.y_pred], _ = theano.scan(fn = forwardProp,
                                                sequences = self.X,
                                                outputs_info = [self.h0, None]
                                            )
                                        
        self.output = T.reshape(self.h, (self.n_in, self.n_out))


        
    def initializeParams(self, n_in, n_hidden, n_out):
        
        # I/P to gate layer params
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_in+n_hidden)),
                            high = np.sqrt(6/(n_in+n_hidden)),
                            size=(n_in, n_hidden)),
                            dtype = datatype)
        W_xg = theano.shared(value = W, name = "W_xg", borrow = True)

        
        # cell state to gate params
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_hidden*2)),
                            high = np.sqrt(6/(n_hidden*2)),
                            size=(n_hidden, n_hidden)),
                            dtype = datatype)        
        
        W_hg = theano.shared(value = W, name = "W_hg", borrow = True)
        
        
        # Gate to O/P params
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_hidden+n_out)),
                            high = np.sqrt(6/(n_hidden+n_out)),
                            size=(n_hidden, n_out)),
                            dtype = datatype)        
        
        W_go = theano.shared(value = W, name = "W_ho", borrow = True)                
        
        h0 = theano.shared(value = np.zeros((n_hidden, ), dtype = datatype), name="h0")
        #Biases
        b_g = theano.shared(value = np.zeros((n_hidden,), dtype = datatype), name="b_g")
        b_o = theano.shared(value = np.zeros((n_hidden,), dtype = datatype), name="b_o")
        
        
        return [W_xg, W_hg, W_go, b_g, b_o, h0]
        
        
        



### Working Version ###
# 1 hidden layer of vanilla RNN
class vanillaRNNNet1(object):
    
    def __init__(self, n_in, n_hidden, n_out, bptt_truncate=4):
        # Assign instance variables
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.bptt_truncate = bptt_truncate
        self.params = self.initializeParam()
        self.W_xh, self.W_hy, self._hh, self.b_h, self.b_y = self.params
        self.theano = {}
        self.__theano_build__()
        
    def initializeParam(self):
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(self.n_in+self.n_hidden)),
                            high = np.sqrt(6/(self.n_in+self.n_hidden)),
                            size=(self.n_hidden, self.n_in)),
                            dtype = datatype)
        W_xh = theano.shared(name='W_xh', value=W, borrow=True)
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(self.n_out+self.n_hidden)),
                            high = np.sqrt(6/(self.n_out+self.n_hidden)),
                            size=(self.n_out, self.n_hidden)),
                            dtype = datatype)
        W_hy = theano.shared(name='W_hy', value=W, borrow=True)
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(self.n_hidden+self.n_hidden)),
                            high = np.sqrt(6/(self.n_hidden+self.n_hidden)),
                            size=(self.n_hidden, self.n_hidden)),
                            dtype = datatype)
        W_hh = theano.shared(name='W_hh', value=W, borrow=True)
        b_h = theano.shared(value = np.zeros((self.n_hidden,), dtype = datatype), name="b_h")
        b_y = theano.shared(value = np.zeros((self.n_out,), dtype = datatype), name="b_y")
        return [W_xh, W_hy, W_hh, b_h, b_y]
        
    
    def __theano_build__(self):
        W_xh, W_hy, W_hh, b_h, b_y = self.params
        x = T.ivector('x')
        y = T.ivector('y')
        def forward_prop_step(x_t, h_t_prev, W_xh, W_hy, W_hh, b_h, b_y):
            h_t = T.tanh(W_xh[:, x_t] + T.dot(W_hh, h_t_prev) + b_h)
            o_t = T.nnet.softmax(T.dot(W_hy, h_t) + b_y)
            return [o_t[0], h_t]
        h_0 = T.zeros(self.n_hidden)
        [o,h], _ = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=h_0)],
            non_sequences=[W_xh, W_hy, W_hh, b_h, b_y],
            truncate_gradient=self.bptt_truncate,
            strict=True)
        
        prediction = T.argmax(o, axis=1)
        
        learning_rate = T.scalar('learning_rate')
        self.cost = T.sum(T.nnet.categorical_crossentropy(o, y))
        self.gparams, self.updates = gradientUpdates(self.cost, self.params, learning_rate)
        

        
        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], self.cost)
        self.bptt = theano.function([x, y], self.gparams)
        
        # SGD
        self.sgd_step = theano.function([x,y,learning_rate], [], 
                      updates=self.updates)
    
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)  
        

