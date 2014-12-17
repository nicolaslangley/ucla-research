import time
import sys
import os
import theano as th
import numpy as np
import theano.tensor as T
from load_mnist_dataset import load_data
from logistic_regression import LogisticRegression


# Function for testing the regression model on the MNIST dataset
def test_regression_model_mnist(dataset_name='mnist.pkl.gz',
                     learning_rate=0.13,
                     n_epochs=1000,
                     batch_size=600):
    # Set up the dataset
    dataset = load_data(dataset_name)
    # Split the data into a training, validation and test set
    train_data, train_labels = dataset[0]
    test_data, test_labels = dataset[1]
    validation_data, validation_labels = dataset[2]
    # Compute number of minibatches for each set
    n_train_batches = train_data.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = validation_data.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_data.get_value(borrow=True).shape[0] / batch_size
    data_dim = (28, 28) # The dimension of each image in the dataset
    data_classes = 10 # The number of classes within the data
    
    # Build the model
    # ---------------

    # Allocate symbolic variables for data
    index = T.lscalar() # This is the index to a minibatch
    x = T.matrix('x') # Data (rasterized images)
    y = T.ivector('y') # Labels (1d vector of ints)

    # Construct logistic regression class
    classifier = LogisticRegression(input=x, n_in=data_dim[0]*data_dim[1], n_out=data_classes)

    # Cost to minimize during training
    cost = classifier.negative_log_likelihood(y)

    # Compile a Theano function that computes mistakes made by the model on a minibatch
    test_model = th.function(inputs=[index], # This function is for the test data   
                             outputs=classifier.errors(y),
                             givens={x: test_data[index * batch_size: (index + 1) * batch_size],
                                     y: test_labels[index * batch_size: (index + 1) * batch_size]})
    validate_model = th.function(inputs=[index], # This function is for the validation data    
                                 outputs=classifier.errors(y),
                                 givens={x: validation_data[index * batch_size: (index + 1) * batch_size],
                                         y: validation_labels[index * batch_size: (index + 1) * batch_size]})
    # Compute the gradient of cost with respect to theta = (W,b)
    grad_W = T.grad(cost=cost, wrt=classifier.W)
    grad_b = T.grad(cost=cost, wrt=classifier.b)

    # Specify how to update model parameters as a list of (variable, update expression) pairs
    updates = [(classifier.W, classifier.W - learning_rate * grad_W),
               (classifier.b, classifier.b - learning_rate * grad_b)]

    # Compile Theano function that returns the cost and updates parameters of model based on update rules
    train_model = th.function(inputs=[index], # Index in minibatch that defines x with label y   
                             outputs=cost, # Cost/loss associated with x,y
                             updates=updates,
                             givens={x: train_data[index * batch_size: (index + 1) * batch_size],
                                     y: train_labels[index * batch_size: (index + 1) * batch_size]})

    # Train the model
    # ---------------

    # Setup the early-stopping parameters
    patience = 5000 # Minimum number of examples to examine
    patience_increase = 2 # How much longer to wait once a new best is found
    improvement_threshold = 0.995 # Value of a significant relative improvement
    validation_frequency = min(n_train_batches, patience / 2) # Number of minibatches before validating
    best_validation_loss = np.inf
    test_score = 0
    start_time = time.clock()

    # Setup the training loop
    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # Set the iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                # Compute the zero-one loss on the validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % (epoch,
                                                                             minibatch_index + 1,
                                                                             n_train_batches,
                                                                             this_validation_loss * 100.))
                # Check if current validation score is the best
                if this_validation_loss < best_validation_loss:
                    # Improve the patience is loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    # Test on test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print('epoch %i, minibatch %i/%i, test error of best model %f %%' % (epoch,
                                                                                         minibatch_index + 1,
                                                                                         n_train_batches,
                                                                                         test_score * 100.))
            # Stop the loop if we have exhausted our patience
            if patience <= iter:
                done_looping = True
                break;
    # The loop has ended so record the time it took
    end_time = time.clock()
    # Print out results and timing information
    print('Optimization complete with best validation score of %f %%, with test performance %f %%' % (best_validation_loss * 100.,
                                                                                                      test_score * 100.)) 
    print 'The code ran for %d epochs with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
    test_regression_model_mnist()
