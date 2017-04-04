# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:11:35 2017

@author: mrins
"""

import theano
import theano.tensor as T
from collections import OrderedDict


def gradientUpdates(cost, params, learningRate):
        
#    assert learningRate > 0 and learningRate < 1

    updates = []           
    gparams = [T.grad(cost, param) for param in params]
    for param, gparam in zip(params, gparams):
        updates.append((param, param - learningRate * gparam))
    return [gparams, updates]
    
    
def gradientUpdateUsingMomentum(cost, params, learningRate, momentum):
    
    assert learningRate > 0 and learningRate < 1
    assert momentum > 0 and momentum < 1
    
    updates = []
    gparams = [T.grad(cost, param) for param in params]
    for param, gparam in zip(params, gparams):
        velocity = theano.shared(param.get_value(borrow=True) * 0, broadcastable = param.broadcastable)
        updates.append(param, param - learningRate * velocity)
        updates.append((velocity, momentum * velocity + (1 - momentum) * gparam))
    return updates
    

    
def gradientUpdatesWithMaxNormConstraint(cost, params, learningRate, maxNorm = 10, epsilon = 1e-7):
   
    assert learningRate > 0 and learningRate < 1
    
    updates = OrderedDict()
    gparams = [T.grad(cost, param) for param in params] 
    for param, gparam in zip(params, gparams):
        updates[param] = param - learningRate * gparam
        columnNormsOfUpdatedParams = T.sqrt(T.sum(T.sqr(updates[param]), axis=0))
        desiredColumnNormsOfUpdatedParams = T.clip(columnNormsOfUpdatedParams, 0, maxNorm)
        scale = (epsilon + desiredColumnNormsOfUpdatedParams) / (epsilon + columnNormsOfUpdatedParams)
        updates[param] = updates[param] * scale
        
    return updates  