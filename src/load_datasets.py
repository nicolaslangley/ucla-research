import numpy as np
import cv2
import os
import csv
import gzip
import cPickle
import theano as th
import theano.tensor as T
import numpy as np

# Load a file with cPickle
def load(name):
    f = open(name, 'rb')
    data = cPickle.load(f)
    f.close()
    return data

# Function for loading the dataset (images and labels.csv file)
def load_custom_dataset(data_dir):
    images = [data_dir + f for f in os.listdir(data_dir) if f.endswith(".jpg")]
    labels = []
    # Load the labels from a CSV file
    csvfile = open(data_dir + 'labels.csv', 'rb') 
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        labels.append(int(row[0]))
    data = []
    for image in images:
        img = cv2.imread(image)
        # Resize the image to be 100x100
        img_rsz = cv2.resize(img,(50,50))
        gray = cv2.cvtColor(img_rsz,cv2.COLOR_BGR2GRAY)
        gray_flat = flatten_image(gray)
        data.append(gray_flat[0])
    # Perform PCA on the images - TODO: does this add anything / is it necessary?
    component_samples = len(data) / 10
    component_labels = labels[:component_samples]
    component_data = []
    for i in range(component_samples):
        current_sample = data[i:i+10]
        (V,S,mean) = pca(current_sample,1)
        pca_result_flat = [] 
        for j in range(V.shape[0]):
            pca_result_flat.append(V[j][0])    
        component_data.append(pca_result_flat)
    #return (np.array(component_data), np.array(component_labels))
    return (np.array(data), np.array(labels))

def load_mnist_datasets():
    # Load the dataset
    dataset= os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..',
                          'image_datasets',
                          'mnist.pkl.gz')
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return (train_set, valid_set, test_set)

def load_cifar_datasets():
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..',
                               'image_datasets',
                               'CIFAR_100_python')
    # Load the CIFAR-100 training dataset
    cifar_train_dataset = load(os.path.join(dataset_dir,'train'))
    data = cifar_train_dataset['data']
    data = data.astype('float') / 255.
    labels = cifar_train_dataset['coarse_labels']
    # Split data into training and validation sets
    train_data = data[0:40000]
    train_labels = np.array(labels[0:40000])
    valid_data = data[40000:50000]
    valid_labels = np.array(labels[40000:50000])
    # Load the CIFAR-100 test dataset
    cifar_test_dataset = load(os.path.join(dataset_dir,'test'))
    test_data = cifar_test_dataset['data']
    test_data = test_data.astype('float') / 255.
    test_labels = np.array(cifar_test_dataset['coarse_labels'])
    return ((train_data, train_labels),
            (valid_data, valid_labels),
            (test_data, test_labels))

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = th.shared(np.asarray(data_x,
                                    dtype=th.config.floatX),
                                    borrow=borrow)
    shared_y = th.shared(np.asarray(data_y,
                                    dtype=th.config.floatX),
                                    borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_data(dataset):
    if dataset == 'mnist':
        print 'Loading the MNIST dataset...'
        train_set, valid_set, test_set = load_mnist_datasets()
    elif dataset =='cifar-100':
        print 'Loading the CIFAR-100 dataset...'
        train_set, valid_set, test_set = load_cifar_datasets()
    # Convert datasets to shared variables 
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
    
            

