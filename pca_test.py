import cv2
import os
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

# Function for flattening the images
def flatten_image(im):
    s = im.shape[0] * im.shape[1]
    im_flat = im.reshape(1,s)
    return im_flat

def pca(data, num_components=2):
    # Numpy PCA
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

# Load sample cat image
#img = cv2.imread('cat3.jpg')
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Load all of the training images
img_dir = "./images/cats_with_people/"
images = [img_dir + f for f in os.listdir(img_dir)]
dataset1 = []
for image in images:
    img = cv2.imread(image)
    img_rsz = cv2.resize(img,(400,300))
    gray = cv2.cvtColor(img_rsz,cv2.COLOR_BGR2GRAY)
    gray_flat = flatten_image(gray)
    dataset1.append(gray_flat[0])
dataset1 = np.array(dataset1)

# Load all of the test images
img_dir = "./images/cats_alone/"
images = [img_dir + f for f in os.listdir(img_dir)]
dataset2 = []
for image in images:
    img = cv2.imread(image)
    img_rsz = cv2.resize(img,(400,300))
    gray = cv2.cvtColor(img_rsz,cv2.COLOR_BGR2GRAY)
    gray_flat = flatten_image(gray)
    dataset2.append(gray_flat[0])
dataset2 = np.array(dataset2)

# Principal Components Analysis on Image
result1 = pca(dataset1,5)
result2 = pca(dataset2,5)

# Project the Principal Components onto the data
#result = np.dot(evecs.T,data).T

plt.figure()
# Plot the first 2 Principal Components
plt.plot(result1[:,0], result1[:,1], '.')

plt.figure()
# Plot the first 2 Principal Components
plt.plot(result2[:,0], result2[:,1], '.')


plt.show()
