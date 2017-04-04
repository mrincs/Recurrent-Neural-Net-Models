# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:50:16 2017

@author: mrins
"""
import numpy as np
import theano
import theano.tensor as T
from GradUpdates import gradientUpdates

datatype = theano.config.floatX

# 1 hidden layer consisting of GRU units
class GRUNet1(object):
    
    def __init__(self, n_in, n_hidden, n_out, bptt_truncate=4):
        # Assign instance variables
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.bptt_truncate = bptt_truncate
        self.params = self.initializeParam()
        self.W_xz, self.W_xr, self.W_x_hcap, \
            self.W_sz, self.W_sr, self.W_s_hcap, self.W_o, \
            self.b_z, self.b_r, self.b_hcap, self.b_o = self.params
        self.theano = {}
        self.__theano_build__()
        
        
        
    def initializeParam(self):
        n_in, n_hidden, n_out = self.n_in, self.n_hidden, self.n_out
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_in+n_hidden)),
                            high = np.sqrt(6/(n_in+n_hidden)),
                            size=(n_hidden, n_in)),
                            dtype = datatype)
        W_xz = theano.shared(value = W, name = "W_xz", borrow = True)
        W_xr = theano.shared(value = W, name = "W_xr", borrow = True)
        W_x_hcap = theano.shared(value = W, name = "W_x_hcap", borrow = True)
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_hidden*2)),
                            high = np.sqrt(6/(n_hidden*2)),
                            size=(n_hidden, n_hidden)),
                            dtype = datatype)        
        W_sz = theano.shared(value = W, name = "W_sz", borrow = True)
        W_sr = theano.shared(value = W, name = "W_sr", borrow = True)
        W_s_hcap = theano.shared(value = W, name = "W_s_hcap", borrow = True)
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_hidden+n_out)),
                            high = np.sqrt(6/(n_hidden+n_out)),
                            size=(n_out, n_hidden)),
                            dtype = datatype)        
        W_o = theano.shared(value = W, name = "W_o", borrow = True)
        
        #Biases
        b_z = theano.shared(value = np.zeros((n_hidden,), dtype = datatype), name="b_z")
        b_r = theano.shared(value = np.zeros((n_hidden,), dtype = datatype), name="b_r")
        b_hcap = theano.shared(value = np.zeros((n_hidden,), dtype = datatype), name="b_hcap")
        b_o = theano.shared(value = np.zeros((n_out,), dtype = datatype), name="b_o")
        
        return [W_xz, W_xr, W_x_hcap, W_sz, W_sr, W_s_hcap, W_o, b_z, b_r, b_hcap, b_o]

        
    
    def __theano_build__(self):
        W_xz, W_xr, W_x_hcap, W_sz, W_sr, W_s_hcap, W_o, b_z, b_r, b_hcap, b_o = self.params
        x = T.ivector('x')
        y = T.ivector('y')
        def forward_prop_step(x_t, s_t_prev, W_xz,W_xr,W_x_hcap,W_sz,W_sr,W_s_hcap,W_o, b_z,b_r,b_hcap,b_o):
            z_t = T.nnet.sigmoid(W_xz[:, x_t] + T.dot(W_sz, s_t_prev) + b_z)
            r_t = T.nnet.sigmoid(W_xr[:, x_t] + T.dot(W_sr, s_t_prev) + b_r)
            h_t_cap = T.tanh(W_x_hcap[:, x_t] + T.dot(W_s_hcap, (r_t * s_t_prev)) + b_hcap)
            s_t = (T.ones_like(z_t) - z_t) + z_t * h_t_cap
            o_t = T.nnet.softmax(T.dot(W_o, s_t) + b_o)          
            return [o_t[0], s_t]
        h_0 = T.zeros(self.n_hidden)
        [o,h], _ = theano.scan(
            fn=forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=h_0)],
            non_sequences=[W_xz, W_xr, W_x_hcap, W_sz, W_sr, W_s_hcap, W_o, b_z, b_r, b_hcap, b_o],
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
        