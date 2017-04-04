# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 21:01:16 2017

@author: mrins
"""


"""
Low efficiency - Without any optimizations for improving performance
"""
import numpy as np
import theano
import theano.tensor as T
from LogisticRegression import SoftMaxLayer
from GradUpdates import gradientUpdates


datatype = theano.config.floatX


class LSTMLayer(object):
    
    def __init__(self, rng, input, shape, is_train = 1, dropOut_prob = 0, params = None):
        
        self.input = input
        self.n_in, self.n_out = shape
        
        if params == None:
            self.W_xi, self.W_xf, self.W_xo, self.W_xc, \
            self.W_hi, self.W_hf, self.W_ho, self.W_hc, \
            self.b_i, self.b_f, self.b_o, self.b_c = self.initialize_Params(self.n_in, self.n_out)
        
            self.params = [self.W_xi, self.W_xf, self.W_xo, self.W_xc, 
                       self.W_hi, self.W_hf, self.W_ho, self.W_hc,
                       self.b_i, self.b_f, self.b_o, self.b_c]
        else:
            self.W_xi, self.W_xf, self.W_xo, self.W_xc, \
            self.W_hi, self.W_hf, self.W_ho, self.W_hc, \
            self.b_i, self.b_f, self.b_o, self.b_c = params
            
                               
        [self.h, self.c], updates = theano.scan(fn = self.forward_Propagation,
                                        sequences = [self.input],
                                        outputs_info = [T.alloc(datatype(0.), 1, 1),
                                                        T.alloc(datatype(0.), 1, 1)
                                                        ],
                                        non_sequences = self.params,
                                        strict = True
                                        )
                                        
        self.output = T.reshape(self.h, (self.n_in, self.n_out))
            
        # DropOut
        if dropOut_prob > 0:
            training_output = DropOutUnits(rng, self.output, dropOut_prob)
            predict_output = ScaleWeight(self.output, dropOut_prob)
            self.output = T.switch(T.neq(is_train, 0), training_output, predict_output)

        
    def initialize_Params(self, n_in, n_hidden):
        
        # I/P to hidden layer params
        W = np.random.uniform(
                            low = -np.sqrt(6/(n_in+n_hidden)),
                            high = np.sqrt(6/(n_in+n_hidden)),
                            size=(n_in, n_hidden),
                            dtype = datatype)
        W_xi = theano.shared(value = W, name = "W_xi", borrow = True)
        W_xf = theano.shared(value = W, name = "W_xf", borrow = True)
        W_xo = theano.shared(value = W, name = "W_xo", borrow = True)
        W_xc = theano.shared(value = W, name = "W_xc", borrow = True)

        
        # Hidden to hidden layer params
        W = np.random.uniform(
                            low = -np.sqrt(6/(n_hidden*2)),
                            high = np.sqrt(6/(n_hidden*2)),
                            size=(n_hidden, n_hidden),
                            dtype = datatype)        
        
        W_hi = theano.shared(value = W, name = "W_hi", borrow = True)
        W_hf = theano.shared(value = W, name = "W_hf", borrow = True)
        W_ho = theano.shared(value = W, name = "W_ho", borrow = True)
        W_hc = theano.shared(value = W, name = "W_hc", borrow = True)
        
        #Biases
        b_i = theano.shared(value = np.zeros((n_hidden,), dtype = datatype), name="b_i")
        b_f = theano.shared(value = np.zeros((n_hidden,), dtype = datatype), name="b_f")
        b_o = theano.shared(value = np.zeros((n_hidden,), dtype = datatype), name="b_o")
        b_c = theano.shared(value = np.zeros((n_hidden,), dtype = datatype), name="b_c")
        
        return [W_xi, W_xf, W_xo, W_xc, W_hi, W_hf, W_ho, W_hc, b_i, b_f, b_o, b_c]
        
        
        
    def forward_Propagation(self):
        
        i_t = T.nnet.sigmoid(theano.dot(self.W_xi, self.input) + theano.dot(self.W_hi, self.h_tprev) + self.b_i)
        f_t = T.nnet.sigmoid(theano.dot(self.W_xf, self.input) + theano.dot(self.W_hf, self.h_tprev) + self.b_f)
        c_t_cap = T.tanh(theano.dot(self.W_xc, self.input) + theano.dot(self.W_hc, self.h_tprev) + self.b_c)
        c_t = i_t * c_t_cap + f_t * self.c_tprev
        o_t = T.nnet.sigmoid(theano.dot(self.W_xo, self.input) + theano.dot(self.W_ho, self.h_tprev) + self.b_o)
        h_t = o_t * T.tanh(c_t)
        self.h_tprev = h_t
        self.c_tprev = c_t
        return h_t, c_t


# 1 hidden layer consisting of LSTM units
class LSTMNet1(object):
    
    def __init__(self, n_in, n_hidden, n_out, bptt_truncate=4):
        # Assign instance variables
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.bptt_truncate = bptt_truncate
        self.params = self.initializeParam()
        self.W_xi, self.W_xf, self.W_xo, self.W_xc, \
            self.W_hi, self.W_hf, self.W_ho, self.W_hc, self.W_y, \
            self.b_i, self.b_f, self.b_o, self.b_c, self.b_y = self.params
        self.theano = {}
        self.__theano_build__()
        
    def initializeParam(self):
        n_in, n_hidden, n_out = self.n_in, self.n_hidden, self.n_out
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_in+n_hidden)),
                            high = np.sqrt(6/(n_in+n_hidden)),
                            size=(n_hidden, n_in)),
                            dtype = datatype)
        W_xi = theano.shared(value = W, name = "W_xi", borrow = True)
        W_xf = theano.shared(value = W, name = "W_xf", borrow = True)
        W_xo = theano.shared(value = W, name = "W_xo", borrow = True)
        W_xc = theano.shared(value = W, name = "W_xc", borrow = True)
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_hidden*2)),
                            high = np.sqrt(6/(n_hidden*2)),
                            size=(n_hidden, n_hidden)),
                            dtype = datatype)        
        W_hi = theano.shared(value = W, name = "W_hi", borrow = True)
        W_hf = theano.shared(value = W, name = "W_hf", borrow = True)
        W_ho = theano.shared(value = W, name = "W_ho", borrow = True)
        W_hc = theano.shared(value = W, name = "W_hc", borrow = True)
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_hidden+n_out)),
                            high = np.sqrt(6/(n_hidden+n_out)),
                            size=(n_out, n_hidden)),
                            dtype = datatype) 
        W_y = theano.shared(value = W, name = "W_y", borrow = True)                   
        #Biases
        b_i = theano.shared(value = np.zeros((n_hidden,), dtype = datatype), name="b_i")
        b_f = theano.shared(value = np.zeros((n_hidden,), dtype = datatype), name="b_f")
        b_o = theano.shared(value = np.zeros((n_hidden,), dtype = datatype), name="b_o")
        b_c = theano.shared(value = np.zeros((n_hidden,), dtype = datatype), name="b_c")
        b_y = theano.shared(value = np.zeros((n_out,), dtype = datatype), name="b_y")
        
        return [W_xi, W_xf, W_xo, W_xc, W_hi, W_hf, W_ho, W_hc, W_y, b_i, b_f, b_o, b_c, b_y]

        
    
    def __theano_build__(self):
        W_xi, W_xf, W_xo, W_xc, W_hi, W_hf, W_ho, W_hc, W_y, b_i, b_f, b_o, b_c, b_y = self.params
        x = T.ivector('x')
        y = T.ivector('y')
        def forward_prop_step(x_t, h_t_prev, c_t_prev, W_xi, W_xf, W_xo, W_xc, W_hi, W_hf, W_ho, W_hc, W_y, b_i, b_f, b_o, b_c, b_y):
            i_t = T.nnet.sigmoid(W_xi[:, x_t] + T.dot(W_hi, h_t_prev) + b_i)
            f_t = T.nnet.sigmoid(W_xf[:, x_t] + T.dot(W_hf, h_t_prev) + b_f)
            o_t = T.nnet.sigmoid(W_xo[:, x_t] + T.dot(W_ho, h_t_prev) + b_o)
            c_t_cap = T.tanh(W_xc[:, x_t] + T.dot(W_hc, h_t_prev) + b_c)
            c_t = i_t * c_t_cap + f_t * c_t_prev
            h_t = o_t * T.tanh(c_t)
            y_t = T.nnet.softmax(T.dot(W_y, h_t) + b_y)
            return [y_t[0], h_t, c_t]
        h_0 = T.zeros(self.n_hidden)
        c_0 = T.zeros(self.n_hidden)
        [o,h,c], _ = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=h_0), dict(initial=c_0)],
            non_sequences=[W_xi, W_xf, W_xo, W_xc, W_hi, W_hf, W_ho, W_hc, W_y, b_i, b_f, b_o, b_c, b_y],
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
        




# 1 hidden layer consisting of LSTM units. Using lesser parameter
# Behavior same as LSTMNet1
class LSTMNet2(object):
    
    def __init__(self, n_in, n_hidden, n_out, bptt_truncate=4):
        # Assign instance variables
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.bptt_truncate = bptt_truncate
        self.params = self.initializeParam()
        self.W_x, self.W_h, self.W_y, self.b, self.b_y = self.params
        self.theano = {}
        self.__theano_build__()
        
    def initializeParam(self):
        n_in, n_hidden, n_out = self.n_in, self.n_hidden, self.n_out
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_in+n_hidden)),
                            high = np.sqrt(6/(n_in+n_hidden)),
                            size=(4, n_hidden, n_in)),
                            dtype = datatype)
        W_x = theano.shared(value = W, name = "W_x", borrow = True)
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_hidden*2)),
                            high = np.sqrt(6/(n_hidden*2)),
                            size=(4, n_hidden, n_hidden)),
                            dtype = datatype)        
        W_h = theano.shared(value = W, name = "W_h", borrow = True)
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_hidden+n_out)),
                            high = np.sqrt(6/(n_hidden+n_out)),
                            size=(n_out, n_hidden)),
                            dtype = datatype) 
        W_y = theano.shared(value = W, name = "W_y", borrow = True)                   
        #Biases
        b = theano.shared(value = np.zeros((4, n_hidden), dtype = datatype), name="b_i")
        b_y = theano.shared(value = np.zeros((n_out,), dtype = datatype), name="b_y")
        
        return [W_x, W_h, W_y, b, b_y]

        
    
    def __theano_build__(self):
        W_x, W_h, W_y, b, b_y = self.params
        x = T.ivector('x')
        y = T.ivector('y')
        def forward_prop_step(x_t, h_t_prev, c_t_prev, W_x, W_h, W_y, b, b_y):
            i_t = T.nnet.sigmoid(W_x[0, :, x_t] + T.dot(W_h[0], h_t_prev) + b[0])
            f_t = T.nnet.sigmoid(W_x[1, :, x_t] + T.dot(W_h[1], h_t_prev) + b[1])
            o_t = T.nnet.sigmoid(W_x[2, :, x_t] + T.dot(W_h[2], h_t_prev) + b[2])
            c_t_cap = T.tanh(W_x[3, :, x_t] + T.dot(W_h[3], h_t_prev) + b[3])
            c_t = i_t * c_t_cap + f_t * c_t_prev
            h_t = o_t * T.tanh(c_t)
            y_t = T.nnet.softmax(T.dot(W_y, h_t) + b_y)
            return [y_t[0], h_t, c_t]
        h_0 = T.zeros(self.n_hidden)
        c_0 = T.zeros(self.n_hidden)
        [o,h,c], _ = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=h_0), dict(initial=c_0)],
            non_sequences=[W_x, W_h, W_y, b, b_y],
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




# 2 hidden layers consisting of LSTM units stacked upon each other
# Improved on top of LSTMNet2
class StackedLSTMNet1(object):
    
    def __init__(self, n_in, n_hidden, n_out, bptt_truncate=4):
        # Assign instance variables
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.bptt_truncate = bptt_truncate
        self.params = self.initializeParam()
        self.E, self.W_x, self.W_h, self.W_y, self.b, self.b_y = self.params
        self.theano = {}
        self.__theano_build__()
        
    def initializeParam(self):
        n_in, n_hidden, n_out = self.n_in, self.n_hidden, self.n_out
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_in+n_hidden)),
                            high = np.sqrt(6/(n_in+n_hidden)),
                            size=(n_hidden, n_in)),
                            dtype = datatype)
        E = theano.shared(value = W, name = "E", borrow = True) # To maintain symmetry
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_in+n_hidden)),
                            high = np.sqrt(6/(n_in+n_hidden)),
                            size=(8, n_hidden, n_hidden)),
                            dtype = datatype)
        W_x = theano.shared(value = W, name = "W_x", borrow = True)
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_hidden*2)),
                            high = np.sqrt(6/(n_hidden*2)),
                            size=(8, n_hidden, n_hidden)),
                            dtype = datatype)        
        W_h = theano.shared(value = W, name = "W_h", borrow = True)
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_hidden+n_out)),
                            high = np.sqrt(6/(n_hidden+n_out)),
                            size=(n_out, n_hidden)),
                            dtype = datatype) 
        W_y = theano.shared(value = W, name = "W_y", borrow = True)                   
        b = theano.shared(value = np.zeros((8, n_hidden), dtype = datatype), name="b_i")
        b_y = theano.shared(value = np.zeros((n_out,), dtype = datatype), name="b_y")
        
        return [E, W_x, W_h, W_y, b, b_y]

        
    
    def __theano_build__(self):
        E, W_x, W_h, W_y, b, b_y = self.params
        x = T.ivector('x')
        y = T.ivector('y')
        
        def forward_prop_step(x_t, h1_t_prev, c1_t_prev, h2_t_prev, c2_t_prev, E, W_x, W_h, W_y, b, b_y):
            
            # Layer 1
            x_layer = E[:, x_t]
            i1_t = T.nnet.sigmoid(T.dot(W_x[0], x_layer) + T.dot(W_h[0], h1_t_prev) + b[0])
            f1_t = T.nnet.sigmoid(T.dot(W_x[1], x_layer) + T.dot(W_h[1], h1_t_prev) + b[1])
            o1_t = T.nnet.sigmoid(T.dot(W_x[2], x_layer) + T.dot(W_h[2], h1_t_prev) + b[2])
            c1_t_cap = T.tanh(T.dot(W_x[3], x_layer) + T.dot(W_h[3], h1_t_prev) + b[3])
            c1_t = i1_t * c1_t_cap + f1_t * c1_t_prev
            h1_t = o1_t * T.tanh(c1_t)
            
            
            # Layer 2
            x_layer = h1_t
            i2_t = T.nnet.sigmoid(T.dot(W_x[4], x_layer) + T.dot(W_h[4], h2_t_prev) + b[4])
            f2_t = T.nnet.sigmoid(T.dot(W_x[5], x_layer) + T.dot(W_h[5], h2_t_prev) + b[5])
            o2_t = T.nnet.sigmoid(T.dot(W_x[6], x_layer) + T.dot(W_h[6], h2_t_prev) + b[6])
            c2_t_cap = T.tanh(T.dot(W_x[7], x_layer) + T.dot(W_h[7], h2_t_prev) + b[7])
            c2_t = i2_t * c2_t_cap + f2_t * c2_t_prev
            h2_t = o2_t * T.tanh(c2_t)
            
            y_t = T.nnet.softmax(T.dot(W_y, h1_t) + b_y)
            return [y_t[0], h1_t, c1_t, h2_t, c2_t]
            
        h1_0 = T.zeros(self.n_hidden)
        c1_0 = T.zeros(self.n_hidden)
        h2_0 = T.zeros(self.n_hidden)
        c2_0 = T.zeros(self.n_hidden)
        [o,h1,c1,h2,c2], _ = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=h1_0), dict(initial=c1_0), dict(initial=h2_0), dict(initial=c2_0)],
            non_sequences=[E, W_x, W_h, W_y, b, b_y],
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



# 1 hidden layer bi-directional LSTM
# Built as an extension of LSTMNet2
class BidirectionalLSTMNet1(object):
    
    def __init__(self, n_in, n_hidden, n_out, bptt_truncate=4):
        # Assign instance variables
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.bptt_truncate = bptt_truncate
        self.params = self.initializeParam()
        self.f_W_x, self.b_W_x, self.f_W_h, self.b_W_h, self.f_W_y, self.b_W_y, self.f_b, self.b_b, self.b_y = self.params
        self.theano = {}
        self.__theano_build__()
        
    def initializeParam(self):
        n_in, n_hidden, n_out = self.n_in, self.n_hidden, self.n_out
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_in+n_hidden)),
                            high = np.sqrt(6/(n_in+n_hidden)),
                            size=(4, n_hidden, n_in)),
                            dtype = datatype)
        f_W_x = theano.shared(value = W, name = "f_W_x", borrow = True)
        b_W_x = theano.shared(value = W, name = "b_W_x", borrow = True)
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_hidden*2)),
                            high = np.sqrt(6/(n_hidden*2)),
                            size=(4, n_hidden, n_hidden)),
                            dtype = datatype)        
        f_W_h = theano.shared(value = W, name = "f_W_h", borrow = True)
        b_W_h = theano.shared(value = W, name = "b_W_h", borrow = True)
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_hidden+n_out)),
                            high = np.sqrt(6/(n_hidden+n_out)),
                            size=(n_out, n_hidden)),
                            dtype = datatype) 
        f_W_y = theano.shared(value = W, name = "f_W_y", borrow = True)    
        b_W_y = theano.shared(value = W, name = "b_W_y", borrow = True)               

        f_b = theano.shared(value = np.zeros((4, n_hidden), dtype = datatype), name="f_b")
        b_b = theano.shared(value = np.zeros((4, n_hidden), dtype = datatype), name="b_b")
        b_y = theano.shared(value = np.zeros((n_out,), dtype = datatype), name="b_y")
        
        return [f_W_x, b_W_x, f_W_h, b_W_h, f_W_y, b_W_y, f_b, b_b, b_y]

        
    
    def __theano_build__(self):
        f_W_x, b_W_x, f_W_h, b_W_h, f_W_y, b_W_y, f_b, b_b, b_y = self.params
        f_x = T.ivector('f_x')
        b_x = f_x[::-1]
        y = T.ivector('y')
        def forward_prop_step(f_x_t, b_x_t, f_h_t_prev, b_h_t_prev, f_c_t_prev, b_c_t_prev, f_W_x, b_W_x, f_W_h, b_W_h, f_W_y, b_W_y, f_b, b_b, b_y):
            
            # Forward LSTM
            f_i_t = T.nnet.sigmoid(f_W_x[0, :, f_x_t] + T.dot(f_W_h[0], f_h_t_prev) + f_b[0])
            f_f_t = T.nnet.sigmoid(f_W_x[1, :, f_x_t] + T.dot(f_W_h[1], f_h_t_prev) + f_b[1])
            f_o_t = T.nnet.sigmoid(f_W_x[2, :, f_x_t] + T.dot(f_W_h[2], f_h_t_prev) + f_b[2])
            f_c_t_cap = T.tanh(f_W_x[3, :, f_x_t] + T.dot(f_W_h[3], f_h_t_prev) + f_b[3])
            f_c_t = f_i_t * f_c_t_cap + f_f_t * f_c_t_prev
            f_h_t = f_o_t * T.tanh(f_c_t)
            
            # Backward LSTM
            b_i_t = T.nnet.sigmoid(b_W_x[0, :, b_x_t] + T.dot(b_W_h[0], b_h_t_prev) + b_b[0])
            b_f_t = T.nnet.sigmoid(b_W_x[1, :, b_x_t] + T.dot(b_W_h[1], b_h_t_prev) + b_b[1])
            b_o_t = T.nnet.sigmoid(b_W_x[2, :, b_x_t] + T.dot(b_W_h[2], b_h_t_prev) + b_b[2])
            b_c_t_cap = T.tanh(b_W_x[3, :, b_x_t] + T.dot(b_W_h[3], b_h_t_prev) + b_b[3])
            b_c_t = b_i_t * b_c_t_cap + b_f_t * b_c_t_prev
            b_h_t = b_o_t * T.tanh(b_c_t)
            
            y_t = T.nnet.softmax(T.dot(f_W_y, f_h_t) + T.dot(b_W_y, b_h_t) + b_y)
            return [y_t[0], f_h_t, f_c_t, b_h_t, b_c_t]
            
        f_h_0 = T.zeros(self.n_hidden)
        f_c_0 = T.zeros(self.n_hidden)
        b_h_0 = T.zeros(self.n_hidden)
        b_c_0 = T.zeros(self.n_hidden)
        [o,fh,fc, bh, bc], _ = theano.scan(
            forward_prop_step,
            sequences=[f_x, b_x],
            outputs_info=[None, dict(initial=f_h_0), dict(initial=f_c_0), dict(initial=b_h_0), dict(initial=b_c_0)],
            non_sequences=[f_W_x, b_W_x, f_W_h, b_W_h, f_W_y, b_W_y, f_b, b_b, b_y],
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



def DropOutUnits(rng, layer, dropOut_prob):
     """ Drop out units from layers during training phase
        'mask' creates binary vector of zeros and ones to suppress any node 
     """
     retain_prob = 1 - dropOut_prob
     srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
     mask = srng.binomial(n=1, p=retain_prob, size=layer.shape, dtype=theano.config.floatX)
     return T.cast(layer * mask, theano.config.floatX)
     
     
def ScaleWeight(layer, dropOut_prob):
    """ Rescale weights for averaging of layers during validation/test phase
    """
    retain_prob = 1 - dropOut_prob
    return T.cast(retain_prob * layer, theano.config.floatX)
    
    
    
    