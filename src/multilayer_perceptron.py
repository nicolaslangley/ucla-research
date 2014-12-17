import os
import sys
import time
import numpy as np
import theano as th
import theano.tensor as T
from logistic_regression import LogisticRegression

# Class for the Hidden Layer of the MLP
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        # Weights of the hidden layer are uniformly sampled from a symmetric interval
        # Weights are dependant on the choice of activation function
        if W is None:
            # For tanh as activation function, this is the interval to use 
            W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                              high=np.sqrt(6. / (n_in + n_out)),
                                              size=(n_in, n_out)),
                                  dtype=th.config.floatX)
            # If the activation function is a sigmoid, multiply the weights by 4
            if activation == T.nnet.sigmoid:
                W_values *= 4
            # Make a shared Theano variable for the weight matrix W
            W = th.shared(value=W_values, name='W', borrow=True)
        # Set the biases to be an array of zeros  
        if b is None:
            b_values = np.zeros((n_out,), dtype=th.config.floatX)
            b = th.shared(value=b_values, name='b', borrow=True)
        
        self.W = W
        self.b = b
        # Compute the output of the activation function for the hidden layer
        # output is h(x) = s(b + Wx) where s is the activation function
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output)) # Apply activation function to lin_output
        # Set the paramters of the model
        self.params = [self.W, self.b]

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        # Create the hidden layer - Note: only dealing with a single layer MLP
        self.hidden_layer = HiddenLayer(rng=rng,
                                        input=input,
                                        n_in=n_in,
                                        n_out=n_hidden,
                                        activation=T.tanh) # Can replace the activation function
        # Create the logistic regression layer on top of the hidden layer
        self.log_regression_layer = LogisticRegression(input=self.hidden_layer.output,
                                                       n_in=n_hidden,
                                                       n_out=n_out)
        # Use L1 and L2 regularization to penalize certain parameter configurations
        # Enforce both the L1 norm and the L2 norm squared to be small
        self.L1 = (abs(self.hidden_layer.W).sum() + abs(self.log_regression_layer.W).sum())
        self.L2_sqr = ((self.hidden_layer.W ** 2).sum() + (self.log_regression_layer.W ** 2).sum())
                   
        # Get the negative log-likelihood and errors from the logistic regression layer
        self.negative_log_likelihood = self.log_regression_layer.negative_log_likelihood
        self.errors = self.log_regression_layer.errors

        # Parameters of the model are the parameters of both the hidden and logistic regression layers
        self.params = self.hidden_layer.params + self.log_regression_layer.params
