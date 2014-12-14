from sklearn.cross_validation import train_test_split
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
        # Resize the image to be 32x32
        img_rsz = cv2.resize(img,(32,32))
        gray = cv2.cvtColor(img_rsz,cv2.COLOR_BGR2GRAY)
        gray_flat = flatten_image(gray)
        data.append(gray_flat[0])
    return (np.array(data), np.array(labels))
    

# Load the dataset of images - they should be flattened grayscale values
# To start we are going to be using the MNIST dataset 
dataset = datasets.fetch_mldata("MNIST Original")
# Load both the training and test datasets
(train_data_cats, train_labels_cats) = load_dataset("./train_images/cat_images/")
(train_data_birds, train_labels_birds) = load_dataset("./train_images/bird_images/")
train_data = np.concatenate((train_data_cats, train_data_birds))
train_labels = np.concatenate((train_labels_cats, train_labels_birds))
(test_data, test_labels) = load_dataset("./test_images/")
# Split the data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(dataset.data / 255.0,
                                                  dataset.target.astype("int0"),
                                                  test_size = 0.33)
# Setup the DBN
dbn = DBN([-1, 300, -1], # The number of nodes in the RBM layers (-1 sets to default)
          learn_rates = 0.3, 
          learn_rate_decays = 0.9,
          epochs = 10,
          verbose = 1)
dbn.fit(train_data, train_labels) 

# Evaluate the DBN
predictions = dbn.predict(test_data)
print classification_report(test_labels, predictions)
