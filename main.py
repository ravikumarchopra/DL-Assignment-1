from util import *
from ffnNetwork import FFNNetwork

# Preparing data
x_train, x_test, y_train, y_test= prep_img_data()

# Running forward pass
ffnn=FFNNetwork(x_train.shape[2], hidden_layers=[3])
ffnn.forward_pass(x_train[0])

