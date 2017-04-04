# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 19:33:23 2017

@author: mrins
"""

import numpy as np
import theano as theano
import theano.tensor as T
import operator
from GradUpdates import gradientUpdates

class RNNTheano(object):
    
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))      
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        U, V, W = self.U, self.V, self.W
        x = T.ivector('x')
        y = T.ivector('y')
        def forward_prop_step(x_t, s_t_prev, U, V, W):
            s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev))
            o_t = T.nnet.softmax(V.dot(s_t))
            return [o_t[0], s_t]
        [o,s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[U, V, W],
            truncate_gradient=self.bptt_truncate,
            strict=True)
        
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        # Gradients
        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)
        
        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU, dV, dW])
        
        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x,y,learning_rate], [], 
                      updates=[(self.U, self.U - learning_rate * dU),
                              (self.V, self.V - learning_rate * dV),
                              (self.W, self.W - learning_rate * dW)])
                              
                                      
    
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)   













        
datatype = theano.config.floatX        

class vanillaRNNNet(object):
    
    def __init__(self, n_in, n_hidden, n_out, lr, bptt_truncate=4):
    
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.bptt_truncate = bptt_truncate
        self.lr = lr
        
        self.W_xh, self.W_hh, self.W_hy, self.b_h, self.b_y = self.initializeParams(n_in, n_hidden, n_out)
        self.params = [self.W_xh, self.W_hh, self.W_hy, self.b_h, self.b_y]
        
        self.theano = {}
        self.__theano_build__()
                                       

    def __theano_build__(self):
        U, W, V, b_h, b_y = self.params
        x = T.ivector('x')
        y = T.ivector('y')
        h_0 = theano.shared(value = np.zeros((self.n_hidden, ), dtype = datatype), name="h0")
        def forwardPropStep(x_t, h_tprev, U, V, W):
            h_t = T.tanh(U[:, x_t] + T.dot(W, h_tprev))
            y_t = T.nnet.softmax(T.dot(V, h_t))
            return [y_t[0], h_t]
        [o, h], _ = theano.scan(
            fn = forwardPropStep,
            sequences = x,
            outputs_info = [None, dict(initial=T.zeros(self.n_hidden))],
            non_sequences = [U, V, W],
            truncate_gradient = self.bptt_truncate,
            strict = True
            )
        y_pred = T.argmax(o, axis=1)
        
        self.cost = T.sum(T.nnet.categorical_crossentropy(o, y))
        dU = T.grad(self.cost, U)
        dW = T.grad(self.cost, W)
        dV = T.grad(self.cost, V)
        learning_rate = T.scalar('learning_rate')

        self.updates = [(U, U - learning_rate * dU),
                   (W, W - learning_rate * dW),
                    (V, V - learning_rate * dV)
                    ]
        
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], y_pred)
        self.ce_error = theano.function([x, y], self.cost)
        self.bptt = theano.function([x, y], [dU, dW, dV])
        self.sgd_step = theano.function([x, y], [],
                                        updates = self.updates)
        
        
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)  

        
    def initializeParams(self, n_in, n_hidden, n_out):
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_in+n_hidden)),
                            high = np.sqrt(6/(n_in+n_hidden)),
                            size=(n_hidden, n_in)),
                            dtype = datatype)
        W_xh = theano.shared(value = W, name = "W_xh", borrow = True)
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_hidden+n_out)),
                            high = np.sqrt(6/(n_hidden+n_out)),
                            size=(n_out, n_hidden)),
                            dtype = datatype)
        W_hy = theano.shared(value = W, name = "W_hy", borrow = True)
        W = np.asarray(np.random.uniform(
                            low = -np.sqrt(6/(n_hidden+n_hidden)),
                            high = np.sqrt(6/(n_hidden+n_hidden)),
                            size=(n_hidden, n_hidden)),
                            dtype = datatype)        
        
        W_hh = theano.shared(value = W, name = "W_hh", borrow = True)
        b_h = theano.shared(value = np.zeros((n_hidden,), dtype = datatype), name="b_h")
        b_y = theano.shared(value = np.zeros((n_out,), dtype = datatype), name="b_y")
        return [W_xh, W_hh, W_hy, b_h, b_y]


def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print("Performing gradient check for parameter", pname, "with size ", np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print("Gradient Check ERROR: parameter=", pname, "ix=", ix)
                print("+h Loss: ", gradplus)
                print("-h Loss: ", gradminus)
                print("Estimated_gradient: ", estimated_gradient)
                print("Backpropagation gradient: ", backprop_gradient)
                print("Relative Error: ", relative_error)
                return 
            it.iternext()
        print("Gradient check for parameter ", pname, "passed")