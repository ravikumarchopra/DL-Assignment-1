import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def load_fashion_mnist_data(split_val=False):

    """ Prepares fashion_mnist data for muti-class classification model """

    # Load data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Flaten image data
    x_train= x_train.reshape(x_train.shape[0], 1, x_train.shape[1] * x_train.shape[2])
    x_test= x_test.reshape(x_test.shape[0], 1, x_test.shape[1] * x_test.shape[2])

    # Convert from integers to floats
    x_train_f = x_train.astype(np.float64)
    x_test_f = x_test.astype(np.float64)

    # Normalize
    x_train_norm= x_train_f/255.0
    x_test_norm= x_test_f/255.0

    # Encode output labels using one hot encoding
    encoder=OneHotEncoder()
    y_train_encoded= encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
    y_test_encoded= encoder.transform(y_test.reshape(-1, 1)).toarray()

    if split_val:
      x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, test_size=0.1, random_state=1)
      return x_train_norm, x_val, x_test_norm, y_train_encoded, y_val, y_test_encoded
    else:
      return x_train_norm, x_test_norm, y_train_encoded, y_test_encoded


def load_mnist_data(split_val=False):

    """ Prepares mnist data for muti-class classification model """

    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flaten image data
    x_train= x_train.reshape(x_train.shape[0], 1, x_train.shape[1] * x_train.shape[2])
    x_test= x_test.reshape(x_test.shape[0], 1, x_test.shape[1] * x_test.shape[2])

    # Convert from integers to floats
    x_train_f = x_train.astype(np.float64)
    x_test_f = x_test.astype(np.float64)

    # Normalize
    x_train_norm= x_train_f/255.0
    x_test_norm= x_test_f/255.0

    # Encode output labels using one hot encoding
    encoder=OneHotEncoder()
    y_train_encoded= encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
    y_test_encoded= encoder.transform(y_test.reshape(-1, 1)).toarray()

    if split_val:
      x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, test_size=0.1, random_state=1)
      return x_train_norm, x_val, x_test_norm, y_train_encoded, y_val, y_test_encoded
    else:
      return x_train_norm, x_test_norm, y_train_encoded, y_test_encoded

