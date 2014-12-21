from convolutional_network import CNN
from load_datasets import load_data

custom_datasets = load_data('custom')
#cifar_datasets = load_data('cifar-100')
batch_size = 1 
convnet = CNN(datasets=custom_datasets, batch_size=batch_size)
convnet.train(2)
test_set_x, test_set_y = custom_datasets[2]
test_score = convnet.test(test_set_x, test_set_y, 500)
classify_result = convnet.classify(test_set_x, batch_size)
classify_result.reshape(10000,)
print 'Test Score Result:'
print test_score
