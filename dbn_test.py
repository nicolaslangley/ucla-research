from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
import cv2
import os
import csv
import cPickle

# Function for flattening the images
def flatten_image(im):
    s = im.shape[0] * im.shape[1]
    im_flat = im.reshape(1,s)
    return im_flat

# Load a file with cPickle
def load(name):
    with open(name, 'rb') as f:
        return cPickle.load(f)

def pca(data, num_components=2):
    # Numpy PCA
    data = np.mat(data)
    m, n = data.shape
    mean = data.mean(axis=0)
    data -= mean 
    R = np.cov(data)
    evals, evecs = np.linalg.eigh(R)
    # Sort eigenvalues in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    # Two dimensions of data
    evecs = evecs[:,:num_components]
    return (evecs, evals, mean) 

# Function for loading the dataset (images and labels.csv file)
def load_dataset(data_dir):
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
    
def load_cifar_dataset():
    # Load the CIFAR-100 training dataset
    cifar_train_dataset = load("./image_datasets/CIFAR_100_python/train")
    train_data = cifar_train_dataset['data']
    train_data = train_data.astype('float') / 255.
    train_labels = np.array(cifar_train_dataset['coarse_labels'])
    # Load the CIFAR-100 test dataset
    cifar_test_dataset = load("./image_datasets/CIFAR_100_python/test")
    test_data = cifar_test_dataset['data']
    test_data = test_data.astype('float') / 255.
    test_labels = np.array(cifar_test_dataset['coarse_labels'])
    return (train_data, train_labels, test_data, test_labels)

def load_custom_datasets():
    # Note, bird data skews the dataset
    # Load both the training and test datasets - they should be flattened grayscale values
    (train_data_cats, train_labels_cats) = load_dataset("./train_images/cat_images/")
    (train_data_birds, train_labels_birds) = load_dataset("./train_images/bird_images/")
    (train_data_houses, train_labels_houses) = load_dataset("./train_images/house_images/")
    train_data = np.concatenate((train_data_cats, train_data_houses))
    train_labels = np.concatenate((train_labels_cats, train_labels_houses))
    # Load the test data
    (test_data_cats, test_labels_cats) = load_dataset("./test_images/cat_images/")
    (test_data_birds, test_labels_birds) = load_dataset("./test_images/bird_images/")
    (test_data_houses, test_labels_houses) = load_dataset("./test_images/house_images/")
    test_data = np.concatenate((test_data_cats, test_data_houses))
    test_labels = np.concatenate((test_labels_cats, test_labels_houses))
    return (train_data, train_labels, test_data, test_labels)

(train_data, train_labels, test_data, test_labels) = load_custom_datasets()

n_feat = train_data.shape[1]
n_targets = train_labels.max() + 1

# Setup the DBN
dbn = DBN([n_feat, n_feat / 3, n_targets],
          learn_rates = 0.1,
          learn_rates_pretrain = 0.005,
          epochs = 10,
          verbose = 1)

# Fit the DBN to the data
dbn.fit(train_data, train_labels) 

# Evaluate the DBN performance on the test_data 
predictions = dbn.predict(test_data)
print 'Classification report: \n', classification_report(test_labels, predictions)
print 'Confusion Matrix: \n', confusion_matrix(test_labels, predictions)
