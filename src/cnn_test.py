from convolutional_network import CNN
from load_datasets import load_data

mnist_datasets = load_data('mnist')
cifar_datasets = load_data('cifar-100')
convnet = CNN(cifar_datasets)
