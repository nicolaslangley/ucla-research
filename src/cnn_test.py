from convolutional_network import CNN
from load_datasets import load_data
import sys

custom_datasets = load_data('custom')
canfar_datasets = load_data('canfar-100')
batch_size = 4
print 'Initializing the CNN'
convnet = CNN(datasets=canfar_datasets, batch_size=batch_size)
epochs = 10
print 'Training the CNN for ' + str(epochs) + ' epochs'
convnet.train(epochs)
test_set_x, test_set_y = canfar_datasets[2]
test_score = convnet.test(test_set_x, test_set_y, 500)
classify_result = convnet.classify(test_set_x, batch_size)
classify_result.reshape(10000,)
print 'Test Score Result:'
print test_score
