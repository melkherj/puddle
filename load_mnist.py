from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def load_mnist(categories=range(10)):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_ixs = np.nonzero(mnist.train.labels[:,categories].max(axis=1)>0)[0]
    test_ixs = np.nonzero(mnist.test.labels[:,categories].max(axis=1)>0)[0]
    X_train= mnist.train.images[train_ixs,:]
    Y_train = mnist.train.labels[train_ixs,:][:,categories]
    X_test = mnist.test.images[test_ixs,:]
    Y_test = mnist.test.labels[test_ixs,:][:,categories]
    return X_train,Y_train,X_test,Y_test
