from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
import cv2
import os
import csv

# Function for flattening the images
def flatten_image(im):
    s = im.shape[0] * im.shape[1]
    im_flat = im.reshape(1,s)
    return im_flat

def pca(data, num_components=2):
    # Numpy PCA
    data = np.mat(data)
    m, n = data.shape
    data -= data.mean(axis=0)
    R = np.cov(data)
    evals, evecs = np.linalg.eigh(R)
    # Sort eigenvalues in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    # Two dimensions of data
    evecs = evecs[:,:num_components]
    return evecs 

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
        # Resize the image to be 200x200
        img_rsz = cv2.resize(img,(200,200))
        gray = cv2.cvtColor(img_rsz,cv2.COLOR_BGR2GRAY)
        gray_flat = flatten_image(gray)
        data.append(gray_flat[0])
    return (np.array(data), np.array(labels))
    

# Load both the training and test datasets - they should be flattened grayscale values
(train_data_cats, train_labels_cats) = load_dataset("./train_images/cat_images/")
#(train_data_birds, train_labels_birds) = load_dataset("./train_images/bird_images/")
(train_data_houses, train_labels_houses) = load_dataset("./train_images/house_images/")
train_data = np.concatenate((train_data_cats, train_data_houses))
train_labels = np.concatenate((train_labels_cats, train_labels_houses))
#train_data = train_data_cats
#train_labels = train_labels_cats
(test_data_cats, test_labels_cats) = load_dataset("./test_images/cat_images/")
#(test_data_birds, test_labels_birds) = load_dataset("./test_images/bird_images/")
(test_data_houses, test_labels_houses) = load_dataset("./test_images/house_images/")
test_data = np.concatenate((test_data_cats, test_data_houses))
test_labels = np.concatenate((test_labels_cats, test_labels_houses))

# Setup the DBN
dbn = DBN([-1, 300, -1], # The number of nodes in the RBM layers (-1 sets to default)
          learn_rates = 0.3, 
          learn_rate_decays = 0.9,
          epochs = 10,
          verbose = 1)
# Fit the DBN to the data
dbn.fit(train_data, train_labels) 

# Evaluate the DBN performance on the test_data 
predictions = dbn.predict(test_data)
print classification_report(test_labels, predictions)
