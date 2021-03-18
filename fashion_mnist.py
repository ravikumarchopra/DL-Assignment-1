from util import *
from ffnNetwork import FFNNetwork
from sklearn.metrics import mean_squared_error, accuracy_score

# Preparing data
x_train, x_test, y_train, y_test= load_fashion_mnist_data()

# Training Model 
network = FFNNetwork(x_train.shape[2], output_size= y_train.shape[1], hidden_layers=[32], act_func='sigmoid', loss_func='ce')
network.fit(x_train, y_train, weight_decay=0, display_loss=True, display_accuracy=True, opt_algo='adam', epochs=2, lr=0.001)

# Finding accuracy for test data
y_preds=ffnn.predict(x_test)
y_preds=np.argmax(y_preds, axis=1)
test=np.argmax(y_test, axis=1)
print(accuracy_score(preds, test))
