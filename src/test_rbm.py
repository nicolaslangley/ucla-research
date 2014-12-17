import time
try:
    import PIL.Image as Image
except ImportError:
    import Image
import numpy as np
import theano as th
import theano.tensor as T
import os
from theano.tensor.shared_randomstreams import RandomStreams
from load_mnist_dataset import load_data
from utils import tile_raster_images
from restricted_boltzmann_machine import RBM

def test_rbm_mnist(learning_rate=0.1,
                   training_epochs=1,
                   dataset_name='mnist.pkl.gz',
                   batch_size=20,
                   n_chains=20,
                   n_samples=10,
                   output_folder='../rbm_plots',
                   n_hidden=500):
    ''' Train and sample from RBM using Theano on MNIST dataset '''
    dataset = load_data(dataset_name)

    train_data, train_labels = dataset[0]
    test_data, test_labels = dataset[2]
    data_dim = (28, 28) # The dimension of each image in the dataset
    data_classes = 10 # The number of classes within the data

    # Compute the number of minibatches for the training, validation and testing phases
    n_train_batches = train_data.get_value(borrow=True).shape[0] / batch_size

    # Symbolic variables for the data
    index = T.lscalar() # Index to a minibatch
    x = T.matrix('x') # Data as rasterized images

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # Initialize storage for the persistent chain
    persistent_chain = th.shared(np.zeros((batch_size, n_hidden),dtype=th.config.floatX),borrow=True)

    # Construct the RBM class
    rbm = RBM(input=x,
              n_visible=data_dim[0]*data_dim[1],
              n_hidden=n_hidden,
              numpy_rng=rng,
              theano_rng=theano_rng)

    # Cost and gradient for one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate, persistent=persistent_chain, k=15)

    # Train the RBM
    # -------------
    
    # Create the output folder if it doesn't exist
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # This function updates the RBM parameters
    train_rbm = th.function([index],
                            cost,
                            updates=updates,
                            givens={x: train_data[index * batch_size: (index + 1) * batch_size]},
                            name='train_rbm')
    plotting_time = 0.
    start_time = time.clock()

    # Go through the training epochs
    for epoch in xrange(training_epochs):
        # Go through training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]
        
        print 'Training epoch %d, cost is ' % epoch, np.mean(mean_cost)

        # Plot the filters
        plotting_start = time.clock()
        # Construct image from the weight matrix
        image = Image.fromarray(tile_raster_images(X=rbm.W.get_value(borrow=True).T,
                                                   img_shape=(28, 28),
                                                   tile_shape=(10, 10),
                                                   tile_spacing=(1, 1)))
        # Save the image
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)

    end_time = time.clock()
    pretraining_time = (end_time - start_time) - plotting_time
    print ('Training took %f minutes' % (pretraining_time / 60.))

    # Sample the RBM
    # --------------

    number_of_test_samples = test_data.get_value(borrow=True).shape[0]
    
    # Pick random test examples to initialize persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = th.shared(np.asarray(test_data.get_value(borrow=True)[test_idx:test_idx + n_chains],
                                                dtype=th.config.floatX))
    # Define a function that performs 'plot_every' steps of Gibbs sampling
    plot_every = 1000
    (
     [presig_hids,hid_mfs,hid_samples,presig_vis,vis_mfs,vis_samples],updates
    ) = th.scan(rbm.gibbs_vhv,
                outputs_info=[None, None, None, None, None, persistent_vis_chain],
                n_steps=plot_every)

    # Add persistent chain variable to updates
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # Function that implements persistent chain
    sample_fn = th.function([],
                            [vis_mfs[-1],vis_samples[-1]],
                            updates=updates,
                            name='sample_fn')

    # Create space to store image for plotting
    image_data = np.zeros((29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8')

    # Generate samples
    for idx in xrange(n_samples):
        vis_mf, vis_sample = sample_fn()
        print ' ... plotting sample ', idx
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(X=vis_mf,
                                                                   img_shape=(28,28),
                                                                   tile_shape=(1, n_chains),
                                                                   tile_spacing=(1, 1))
    # Construct the image
    image = Image.fromarray(image_data)
    image.save('samples.png')
    os.chdir('../src')

if __name__ == '__main__':
    test_rbm_mnist()
            
