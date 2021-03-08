from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt


# Loading data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Plotting images
for i in range(9):
  plt.subplot(330 + 1 + i)
  plt.imshow(x_train[i], cmap=plt.get_cmap('gray'), )
plt.show()