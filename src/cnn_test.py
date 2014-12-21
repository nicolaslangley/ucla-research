from convolutional_network import CNN
from load_datasets import load_data
import sys

def to_latex(a,label='A \\oplus B'):
    sys.stdout.write('\[ '
                 + label
                 + ' = \\left| \\begin{array}{' 
                 + ('c'*a.shape[1]) 
                 + '}\n' )
    for r in a:
        sys.stdout.write(str(r[0]))
        for c in r[1:]:
            sys.stdout.write(' & '+str(c))
        sys.stdout.write('\\\\\n')
    sys.stdout.write('\\end{array} \\right| \]\n')

custom_datasets = load_data('custom')
mnist_datasets = load_data('mnist')
batch_size = 500 
print 'Initializing the CNN'
convnet = CNN(datasets=mnist_datasets, batch_size=batch_size)
epochs = 10
print 'Training the CNN for ' + str(epochs) + ' epochs'
convnet.train(epochs)
to_latex(convnet.params[0].get_value(borrow=True))
test_set_x, test_set_y = mnist_datasets[2]
test_score = convnet.test(test_set_x, test_set_y, 500)
classify_result = convnet.classify(test_set_x, batch_size)
classify_result.reshape(10000,)
print 'Test Score Result:'
print test_score
