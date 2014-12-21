import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from logistic_regression import LogisticRegression
from multilayer_perceptron import HiddenLayer

class ConvPoolLayer(object):
    ''' Convolution and Max-Pooling Layer of a LeNet convolutional network '''
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2)):
        # Ensure image and filter have same size
        assert image_shape[1] == filter_shape[1]
        self.input = input
        
        # Set the number of inputs to hidden layer
        # There are "num_input_feature_maps * filter-height * filter_width" inputs to hidden layer 
        fan_in = np.prod(filter_shape[1:])
        # Lower layer gets gradient from "inputs / pooling size"
        fan_out = filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize)

        # Initialize random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = th.shared(np.asarray(rng.uniform(low=-W_bound,
                                                  high=W_bound,
                                                  size=filter_shape),
                                      dtype=th.config.floatX),
                           borrow=True)

        # Bias is a 1d tensor with one bias per ouput feature map
        b_values = np.zeros((filter_shape[0],), dtype=th.config.floatX)
        self.b = th.shared(value=b_values, borrow=True)

        # Convolve input feature maps with the filters
        conv_out = conv.conv2d(input=input,
                               filters=self.W,
                               filter_shape=filter_shape,
                               image_shape=image_shape)

        # Downsample each feature map using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize,
                                            ignore_border=True)
        # Broadcast bias across mini-batches and feature map
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        # Store the model parameters
        self.params = [self.W, self.b]

class CNN(object):
    ''' 
        Convolutional Neural Network with 2 convolutional pooling layers
        NOTE: Dataset is required to be 28x28 images with three sub data sets 
    '''
    def __init__(self, datasets, batch_size=500, nkerns=[20, 50], img_size=(28, 28), learning_rate=0.1):
        
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        
        self.batch_size = batch_size
        # compute number of minibatches for training, validation and testing
        self.n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        self.n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        self.n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        self.n_train_batches /= batch_size
        self.n_valid_batches /= batch_size
        self.n_test_batches /= batch_size

        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        rng = np.random.RandomState(23455)
        
        layer0_input = self.x.reshape((batch_size, 1, img_size[0], img_size[1]))
        
        # Create the two convolutional layers that also perform downsampling using maxpooling
        self.layer0 = ConvPoolLayer(rng,
                                    input=layer0_input,
                                    image_shape=(batch_size, 1, img_size[0], img_size[1]),
                                    filter_shape=(nkerns[0], 1, 5, 5), 
                                    poolsize=(2,2))

        self.layer1 = ConvPoolLayer(rng,
                                    input=self.layer0.output,
                                    image_shape=(batch_size, nkerns[0], 12, 12),
                                    filter_shape=(nkerns[1], nkerns[0], 5, 5), 
                                    poolsize=(2,2))

        layer2_input = self.layer1.output.flatten(2)
       
        # Create the hidden layer of the MLP
        self.layer2 = HiddenLayer(rng,
                                  input=layer2_input,
                                  n_in=nkerns[1] * 4 * 4,
                                  n_out=500,
                                  activation=T.tanh)

        # Create the logistic regression layer for classifiying the results
        self.layer3 = LogisticRegression(input=self.layer2.output, n_in=500, n_out=10)

        self.cost = self.layer3.negative_log_likelihood(self.y)

        self.params = self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params

        self.grads = T.grad(self.cost, self.params)

        # Update list for the paramters to be used when training the model
        updates = [(param_i, param_i - learning_rate * grad_i)
                   for param_i, grad_i in zip(self.params, self.grads)]

        # This function updates the model parameters using Stochastic Gradient Descent
        self.train_model = th.function([self.index],
                                       self.cost, # This is the negative-log-likelihood of the Logisitc Regression layer
                                       updates=updates,
                                       givens={self.x: test_set_x[self.index * batch_size: (self.index + 1) * batch_size],
                                               self.y: test_set_y[self.index * batch_size: (self.index + 1) * batch_size]})
                                     
        # These are Theano functions for testing performance on our test and validation datasets
        self.test_model = th.function([self.index],
                                      self.layer3.errors(self.y),
                                      givens={self.x: test_set_x[self.index * batch_size: (self.index + 1) * batch_size],
                                              self.y: test_set_y[self.index * batch_size: (self.index + 1) * batch_size]})

        self.validate_model = th.function([self.index],
                                          self.layer3.errors(self.y),
                                          givens={self.x: valid_set_x[self.index * batch_size: (self.index + 1) * batch_size],
                                                  self.y: valid_set_y[self.index * batch_size: (self.index + 1) * batch_size]})

    def train(self, n_epochs, patience=10000, patience_increase=2, improvement_threshold=0.995):
        ''' Train the CNN on the training data for a defined number of epochs '''
        # Setup the variables for training the model
        n_train_batches = self.n_train_batches
        n_valid_batches = self.n_valid_batches
        n_test_batches = self.n_test_batches
        validation_frequency = min(n_train_batches, patience / 2)
        best_validation_loss = np.inf
        best_iter = 0
        best_score = 0.
        epoch = 0
        done_looping = False
        # Train the CNN for a defined number of epochs
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):
                iter = (epoch - 1) * n_train_batches + minibatch_index
                # Every 100 iterations
                if iter % 100 == 0:
                    print 'Training iteration ', iter
                    cost_ij = self.train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:
                    # Compute zero-one loss on validation set
                    validation_losses = [self.validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # Check if current validation loss is best so far
                    if this_validation_loss < best_validation_loss:
                        # Improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iter * patience_increase)
                        # Save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter
                if patience <= iter:
                    done_looping = True
                    break
        print 'Optimization complete.'
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    def test(self, set_x, set_y):
        ''' Test data sets and return the test score '''
        # allocate symbolic variables for the data
        n_test_batches = set_x.get_value(borrow=True).shape[0]
        n_test_batches /= self.batch_size
        test_model = th.function(inputs=[self.index],
                                 outputs=self.layer3.errors(self.y),
                                 givens={self.x: set_x[self.index * batch_size: (self.index + 1) * batch_size],
                                         self.y: set_y[self.index * batch_size: (self.index + 1) * batch_size]})
        test_losses = [test_model(i)
                       for i in xrange(n_test_batches)]
        test_score = np.mean(test_losses)
        return test_score

    def classify(self, set):
        ''' 
           Return the labels for the given set
           NOTE: The batch size must be the same as the training set  
        '''
        n_test_batches = set.get_value(borrow=True).shape[0]
        n_test_batches /= self.batch_size
        classify_data = th.function(inputs=[self.index], # Input to this function is a mini-batch at index
                                    outputs=self.layer3.y_pred, # Output the y_predictions
                                    givens={self.x: set[self.index * batch_size: (self.index + 1) * batch_size]}) 
        # Generate labels for the given data
        labels = [classify_data(i)
                  for i in xrange(n_test_batches)]
        return np.array(labels)

