import numpy as np
import theano as th
# Use OpenCV for image resizing
import cv2

def whitening_transform(X):
    ''' Whiten the image - decorrelate and variance to 1 '''
    # Subtract the mean from X
    X_mean = np.mean(X)
    X_norm = X / X_mean
    # Get covariance matrix X^T X
    cov = np.dot(X_norm.T,X_norm)
    # Get eigendecomposition of covariance matrix
    d, V = np.linalg.eigh(cov)
    D = np.diag(1. / np.sqrt(d))
    # Whitening matrix
    W = np.dot(np.dot(V, D), V.T)
    X_white = np.dot(X, W)
    return X_white, W

# Function for loading the dataset (images and labels.csv file)
def load_custom_data(data_dir):
    ''' Load the dataset from data_dir '''
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
        # Resize the image to be 28x28
        img_rsz = cv2.resize(img_white,(28,28))
        gray = cv2.cvtColor(img_rsz,cv2.COLOR_BGR2GRAY)
        # Whiten the image before flatt 
        img_white = whitening_transform(img)
        gray_flat = flatten_image(gray)
        data.append(gray_flat[0])
    return [data, labels]

def save_dataset(save_dir, data):
    ''' Save the dataset as a .pkl.gz file '''

if __name__ == '__main__':
    # This is the directory of the data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..',
                            'image_datasets',
                            'custom_dataset')
    # Load the different datasets     
    train_dataset = load_custom_data(os.path.join(dataset_dir,'train'))
    valid_dataset = load_custom_data(os.path.join(dataset_dir,'train'))
    test_dataset = load_custom_data(os.path.join(dataset_dir,'test'))

    # Save data to a file
    data = [train_dataset, valid_dataset, test_dataset]
    save_dataset(data_dir, data)    


