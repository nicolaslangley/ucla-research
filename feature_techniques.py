import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load sample cat image
img = cv2.imread('cat3.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Basic SIFT Keypoints (Features)
sift = cv2.SIFT()
kp_sift = sift.detect(gray,None)
img_sift = cv2.drawKeypoints(gray,kp_sift)
plt.figure()
plt.imshow(img_sift)
plt.title("SIFT Features")

# Basic ORB Keypoints
orb = cv2.ORB()
kp_orb = orb.detect(gray,None)
img_orb = cv2.drawKeypoints(gray,kp_orb,color=(0,255,0),flags=0)
plt.figure()
plt.imshow(img_orb)
plt.title("ORB Features")

# Shi-Tomasi Corner Detector & Good Features to Track
corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)
img_corners = img
for i in corners:
    x,y = i.ravel()
    cv2.circle(img_corners,(x,y),3,255,-1)
plt.figure()
plt.imshow(img_corners)
plt.title("Shi-Tomasi Corners")

# Principal Components Analysis on Image
data = gray
# Numpy PCA
m, n = data.shape
data -= data.mean(axis=0)
R = np.cov(data, rowvar=False)
evals, evecs = np.linalg.eigh(R)
# Sort eigenvalues in decreasing order
idx = np.argsort(evals)[::-1]
evecs = evecs[:,idx]
evals = evals[idx]
# Two dimensions of data
evecs = evecs[:,:2]
# Project the Principal Components onto the data
result = np.dot(evecs.T,data.T).T
plt.figure()
# Plot the first 2 Principal Components
plt.plot(result[:,0], result[:,1], '.')

plt.show()
