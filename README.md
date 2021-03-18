# Feed Forward Neural Network

This repository contains a python implementation of Feed Forward Neural Network with Backpropagation, along with the example scripts for training the network to classify images from ` mnist ` and ` fashion_mnist ` datasets from `keras`.  

# Usage

To use the Neural Network first import ` FFNNetwork ` class from ` ffnNetwork.py ` : 

```pyhton
    from ffnNetwork import FFNNetwork
```

You can now create an instance of the FFNNetwork class. The constructor takes six parameters:

-`input_size :` An integer representing the number of features of input data.  
-`output_size :` An integer representing the number of output classes  ` e.g. 1(default) `.
-`hidden_layers :` A list of integers representing the number of neurons in each layer ` e.g. [16, 32, 128] `.
-`init_func :` A string representing the name of initialization function ` e.g. 'random', 'xavier'(default), 'zero', 'he' ` you want to use to initialize weights and biases.
-`act_func :` A string representing the name of activation function you want to use for neuron ` e.g. 'sigmoid'(default), 'tanh', 'relu' `.
-`loss_func :` A string representing the name of loss function ` e.g. 'mse', 'ce'(default) ` you want to use for calculating the loss and then finding gradients with respect to weights and biases to minimize the loss. Here, `ce` stands for Cross Entropy Loss and `mse` Mean Squared Error.

For example :

```python
    network = FFNNetwork(10, output_size= 4, hidden_layers=[16, 32], act_func='sigmoid', loss_func='ce')
```

The above line of code will create a neural network with 4 layers, containing a layer of 10 input neurons, followed by two hidden layers of 16 and 32 neurons respectively, followed by a layer of 4 output neurons and using `sigmoid` as activation function and `ce` as loss function.

Note that `input_size` must be greater than or equal to 2, and the number of neurons in each layer must be greater than or equal to 1.

# Feeding Forward

To calculate the output of the network when it is given a certain set of inputs, use the forward_pass method. This method takes a single parameter, inputs, which is a list of floats. The number of elements in inputs must be equal to the number of input neurons in the network. The method returns a list of floats representing the output of the network. For example, if network is a neural network with 4 input neurons, we could use the forward_pass method as follows:

```pyhton
    output = network.forward_pass([1.0, 0.23, 0.5, 0.03])
```

# Training

You can train the neural network using the fit method. This method takes 9 parameters:

-`inputs :` A list of lists. The inner lists consist of floats, representing a single set of inputs to the neural network ` e.g. [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]] `.
-`output_labels :` A list of lists. The inner lists consist of floats, representing a single set of expected outputs from the neural network ` e.g. ["T-shirt/top","Trouser","Pullover","Dress","Coat", "Sandal","Shirt","Sneaker","Bag","Ankle boot"] `.
-`epochs :` An integer representing the number of times you want to go through the `inputs`  ` e.g. 1(default) `.
-`lr :` A float which dictates how fast the network should learn ` e.g. 0.001(default) `.
-`weight_decay :` A float representing the L2 regularization factor.
-`display_loss :` A boolean, by pushing it to True we can plot the loss ` e.g. False(default) `. 
-`display_accuracy :` A boolean, by pushing it to True we can print the accuracy ` e.g. False(default) `.
-`opt_algo :` A string representing the name of optimization algorithm you want to use ` e.g.'sgd', 'momentum', 'nestrov', 'rmsprop', 'adam'(default), 'nadam' `.
-`batch_size :` An integer representing after how seeing how many inputs you want to update the Weights and biases ` e.g. 128(default) `.

The number of elements in inputs and output_labels must be equal. The learning rate must be a positive number. Each of the inner lists in inputs must have a number of elements equal to the number of input neurons in the network. Similarly, each of the inner lists in expected_set must have a number of elements equal to the number of output neurons in the network.

Usage of the fit method is shown in the example below:

```python
    {
        network = FFNNetwork(2, output_size= 1, hidden_layers=[16, 32], act_func='sigmoid', loss_func='ce')
        network.fit([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], [0, 1, 2], weight_decay=0, display_loss=True, display_accuracy=True, opt_algo='adam', epochs=10, lr=0.001)
    }
```

Alternatively, you can train the neural network by loading the data from ` keras.datasets `` e.g. mnist, fashion_mnist `, with the ` load_mnist_data ` and ` load_fashion_mnist_data ` methods respectively. 

For using the ` load_mnist_data ` and ` load_fashion_mnist_data ` methods first import ` util.py `:

` from util import -`

Then call the ` load_mnist_data ` or ` load_fashion_mnist_data ` method to load data:

`x_train, x_test, y_train, y_test= load_fashion_mnist_data()`

# Testing Model Accuracy

you can now test the accuracy of you model using the following code:

```pyhton
    {
        y_preds=ffnn.predict(x_test)
        y_preds=np.argmax(y_preds, axis=1)
        test=np.argmax(y_test, axis=1)
        print(accuracy_score(preds, test))
    }    
```

# Examples

## MNIST handwritten digit classification
` mnist.py ` shows how to create and train a neural network which identifies the handwritten digit in the image. This script creates a network with ` 28x28 = 784 ` input neurons and `10` output neuron. The inputs contains handwritten digit images ` 28x28 pixels ` each where image is represented as a ` 28x28 ` matrix where each element of this matrix represents the `rgb` value of corresponding pixel in the image. The output of the network should be a digit corresponding to the input image. The script trains the network using the network with 54000 sample images of handwritten digits. It then tests the accuracy on test data.

## FASHAION-MNIST clothing classification

` fashion_mnist.py ` shows how to create and train a neural network which identifies the clothing in the image. This script creates a network with ` 28x28 = 784 ` input neurons and `10` output neuron. The inputs contains clothing images ` 28x28 pixels ` each where image is represented as a ` 28x28 ` matrix where each element of this matrix represents the `rgb` value of corresponding pixel in the image. The output of the network should be a class corresponding to the input image. The script trains the network using the network with 54000 sample images of clothing. It then tests the accuracy on test data.

