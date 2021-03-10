import numpy as np
from keras.datasets import fashion_mnist
from sklearn.preprocessing import OneHotEncoder

def prep_img_data():

    """ Prepares image data for muti-class classification model """

    # Load data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Flaten image data
    x_train= x_train.reshape(x_train.shape[0], 1, x_train.shape[1] * x_train.shape[2])
    x_test= x_test.reshape(x_test.shape[0], 1, x_test.shape[1] * x_test.shape[2])

    # Convert from integers to floats
    x_train_f = x_train.astype('float32')
    x_test_f = x_test.astype('float32')

    # Normalize
    x_train_norm= x_train_f/255.0
    x_test_norm= x_test_f/255.0

    # Encode output labels using one hot encoding
    encoder=OneHotEncoder()
    y_train_encoded= encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
    y_test_encoded= encoder.transform(y_test.reshape(-1, 1)).toarray()

    return x_train_norm, x_test_norm, y_train_encoded, y_test_encoded
