import numpy as np
import theano as th
import theano.tensor as T

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        # Initialize the weight matrix W with zeros
        self.W = th.shared(value=np.zeros((n_in,n_out),dtype=th.config.floatX),
                           name='W',
                           borrow=True)
        # Initialize the biases b as a vector of zeros
        self.b = th.shared(value=np.zeros((n_out,),dtype=th.config.floatX),
                           name='b',
                           borrow=True)
        # Symbolic expresion for computing the probability matrix of class-membership
        # i.e. P(Y|x,W,b)
        self.py_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        # Symbolic expression for computing the prediction of the class whose probability is maximal
        self.y_pred = T.argmax(self.py_given_x, axis=1)
        # Set the parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        # Define the likelihood
        log_likelihood_probs = T.log(self.py_given_x)
        minibatch_examples = T.arange(y.shape[0])
        # Return mean log-likelihood across the minibatch
        return -T.mean(log_likelihood_probs[minibatch_examples, y])

    def errors(self, y):
        # Check that y has the same dimensions as y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # Check that y has the right datatype
        if y.dtype.startswith('int'):
            # Return any mistakes in prediction (a vector of 1s and 0s)
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
