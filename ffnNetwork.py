import numpy as np

class FFNNetwork:

    def __init__(self, input_size, output_size=1, hidden_layers=[2], init_func='xavier', act_func='sigmoid', loss_func='ce'):

      # Initializing Network Parameters
      self.x= input_size
      self.y= output_size
      self.h= len(hidden_layers)
      self.init_func= init_func
      self.act_func= act_func
      self.loss_func= loss_func
      self.sizes= [self.x] + hidden_layers + [self.y]
      self.W= {}
      self.B= {}
      np.random.seed(0)

      # He Initialization
      if self.init_func=='he':
        for i in range(self.h+1):
          self.W[i+1]=np.random.randn(self.sizes[i], self.sizes[i+1]) * np.sqrt(2 / self.sizes[i-1])
          self.B[i+1]=np.random.randn(1, self.sizes[i+1])
      
      # Xavier Initialization
      elif self.init_func=='xavier':
        for i in range(self.h+1):
          self.W[i+1]=np.random.randn(self.sizes[i], self.sizes[i+1]) * np.sqrt(1 / self.sizes[i-1])
          self.B[i+1]=np.random.randn(1, self.sizes[i+1])

      # Zero Initialization
      elif self.init_func=='zero':
        for i in range(self.h+1):
          self.W[i+1]=np.zeros((self.sizes[i], self.sizes[i+1]))
          self.B[i+1]=np.zeros((1, self.sizes[i+1]))
      
      # Random Initialization
      else:
        for i in range(self.h+1):
          self.W[i+1]=np.random.randn(self.sizes[i], self.sizes[i+1])
          self.B[i+1]=np.random.randn(1, self.sizes[i+1])

    def perceptron(self, x, w, b):
      """ It computes the perceptron output for the inputs passed """
      return np.dot(x, w)+ b

    def activation(self, x):
      """ It computes the activation function used """
      # Using tanh function as the activation function
      if self.act_func == 'tanh':
        return np.tanh(x)
      
      # Using ReLU function as the activation function
      elif self.act_func == 'relu':
        return np.maximum(0, x)

      # Using sigmoid function as the activation function
      else:
        return 1.0/(1.0 + np.exp(-x))

    def grad_activation(self, x):
      """ It computes the gradient of activation function used """
      # Gradient of tanh function
      if self.act_func == 'tanh':
        return (1 - np.square(x))
      
      # Gradient of ReLU function
      elif self.act_func == 'relu':
        return (1.0 * (x>0))

      # Gradient of sigmoid function
      else:
        return x * (1-x)
      

    def softmax(self, y):
      """ It computes the softmax of input array passed """
      max=np.max(y)
      exps=np.exp(y-max)
      return np.exp(y-max)/np.sum(exps)

    def forward_pass(self, x):
      """ It runs a forward pass of Feed Forward Neural Network """
      self.A={}
      self.H={}
      self.H[0]= x
      
      for i in range(self.h+1):
        self.A[i+1]=np.matmul(self.H[i], self.W[i+1]) + self.B[i+1]
        self.H[i+1]=self.activation(self.A[i+1])
      self.H[self.h+1]=self.softmax(self.A[self.h+1])
      return self.H[self.h+1]

    def grad(self, x, y):
      """ It computes the gradient using backpropgation """
      
      L=self.h+1
      self.forward_pass(x)

      self.dW={}
      self.dB={}
      self.dA={}
      self.dH={}

      if self.loss_func=='mse':
        self.dA[L]=(self.H[L]-y) * self.H[L] * (1-self.H[L])
      else:
        self.dA[L]=(self.H[L]-y)

      for k in range(L, 0, -1):
        self.dW[k]=np.matmul(self.H[k-1].T, self.dA[k])
        self.dB[k]=self.dA[k]
        self.dH[k-1]=np.matmul(self.dA[k], self.W[k].T)
        self.dA[k-1]=np.multiply(self.dH[k-1], self.grad_activation(self.H[k-1]))

    def predict(self, X):
      """ It predicts the output for the inputs passed """
      preds=[]
      for x in X:
        preds.append(self.forward_pass(x)) 
      return np.array(preds).squeeze()

    def fit(self, inputs, output_labels, epochs=1, lr=0.001, weight_decay=0, display_loss=False, display_accuracy=False, opt_algo='adam', batch_size=128, gamma=0.9, beta=0.9, beta1=0.9, beta2=0.999, epsi=1e-8):
      """ It trains the Neural Netword """
      
      x_train, x_val, y_train, y_val= train_test_split(inputs, output_labels, test_size=0.1, random_state=1)

      accuracy, val_accuracy, loss, val_loss= {}, {}, {}, {}
      vW, vB, best_W, best_B= {}, {}, {}, {}
      max_accuracy, max_val_accuracy = 0, 0
      for i in range(self.h+1):
        vW[i+1]=np.zeros((self.sizes[i], self.sizes[i+1]))
        vB[i+1]=np.zeros((1, self.sizes[i+1]))

      m=inputs.shape[2]

      for e in tqdm(range(epochs), total=epochs, unit="epoch"):
        dW, dB= {}, {}

        for i in range(self.h+1):
          dW[i+1]=np.zeros((self.sizes[i], self.sizes[i+1]))
          dB[i+1]=np.zeros((1, self.sizes[i+1]))

        # Gradient Descent/ Batch Gradient Descent/ Vanilla Gradient Descent
        if opt_algo=='gd' or opt_algo=='bgd' or opt_algo=='vgd': 

          for x, y in zip(x_train, y_train):
            self.grad(x, y)
            for i in range(self.h+1):
              if l2_regularisation:
                dW[i+1]+=self.dW[i+1] + weight_decay * self.W[i+1]
              else:
                dW[i+1]+=self.dW[i+1]
              dB[i+1]+=self.dB[i+1]

          for i in range(self.h+1):
            self.W[i+1]-= lr* (dW[i+1])/m
            self.B[i+1]-= lr* (dB[i+1])/m

        # Stochastic Gradient Descent
        elif opt_algo=='sgd':
          
          sample_count= 0
          for x, y in zip(x_train, y_train):
            self.grad(x, y)
            sample_count+=1
            for i in range(self.h+1): 
              dW[i+1]+= self.dW[i+1] + weight_decay * self.W[i+1]
              dB[i+1]+= self.dB[i+1]

              if sample_count % batch_size == 0:
                self.W[i+1]-= lr*dW[i+1]/batch_size
                self.B[i+1]-= lr*dB[i+1]/batch_size

        # Momentum Based Gradient Descent
        elif opt_algo=='momentum':

          sample_count=0
          for x, y in zip(x_train, y_train):
            self.grad(x, y)
            for i in range(self.h+1): 
              dW[i+1]+= self.dW[i+1] + weight_decay * self.W[i+1]
              dB[i+1]+= self.dB[i+1]
            
            sample_count+=1
            if sample_count % batch_size == 0:
              for i in range(self.h+1):
                vW[i+1]= (gamma * vW[i+1] + lr*dW[i+1])/batch_size
                vB[i+1]= (gamma * vB[i+1] + lr*dB[i+1])/batch_size
                self.W[i+1]-= vW[i+1]
                self.B[i+1]-= vB[i+1]

        # Nestrov Accelerated Gradient Descent
        elif opt_algo=='nesterov':

          sample_count=0
          for x, y in zip(x_train, y_train):
            self.grad(x, y)
            sample_count+=1
            for i in range(self.h+1): 
              dW[i+1]+= self.dW[i+1] + weight_decay * self.W[i+1]
              dB[i+1]+= self.dB[i+1]

            if sample_count % batch_size == 0:
              tW, tB= {}, {}
              for i in range(self.h+1):
                tW[i+1]= self.W[i+1] - gamma * vW[i+1]
                tB[i+1]= self.B[i+1] - gamma * vB[i+1]
                self.W[i+1]= tW[i+1]
                self.B[i+1]= tB[i+1]

              self.grad(x, y)
              for i in range(self.h+1):
                vW[i+1]= gamma * vW[i+1] + lr * self.dW[i+1]
                vB[i+1]= gamma * vB[i+1] + lr * self.dB[i+1]
                self.W[i+1]= tW[i+1] - vW[i+1]
                self.B[i+1]= tB[i+1] - vB[i+1]
                               
        # RMSProp Gradient Descent
        elif opt_algo=='rmsprop':

          sample_count=0
          for x, y in zip(x_train, y_train):
            self.grad(x, y)
            sample_count+=1
            for i in range(self.h+1): 
              dW[i+1]+= self.dW[i+1] + weight_decay * self.W[i+1]
              dB[i+1]+= self.dB[i+1]

            if sample_count % batch_size == 0:

              for i in range(self.h+1):
                vW[i+1]= beta * vW[i+1] + (1-beta) * np.power(dW[i+1], 2)
                vB[i+1]= beta * vB[i+1] + (1-beta) * np.power(dB[i+1], 2)
                self.W[i+1]-= (lr/np.sqrt(vW[i+1] + epsi)) * dW[i+1]
                self.B[i+1]-= (lr/np.sqrt(vB[i+1] + epsi)) * dB[i+1]                

        # Adam Gradient Descent
        elif opt_algo=='adam':

          sample_count=0
          for x, y in zip(x_train, y_train):
            self.grad(x, y)
            sample_count+=1
            for i in range(self.h+1): 
              dW[i+1]+= self.dW[i+1] + weight_decay * self.W[i+1]
              dB[i+1]+= self.dB[i+1]

            if sample_count % batch_size == 0:
              mW, mB= {}, {}
              for i in range(self.h+1):
                mW[i+1]= np.zeros(dW[i+1].shape)
                mB[i+1]= np.zeros(dB[i+1].shape)
                mW[i+1]= beta1 * mW[i+1] + (1 - beta1) * dW[i+1]
                mB[i+1]= beta1 * mB[i+1] + (1 - beta1) * dB[i+1]

                vW[i+1]= beta2 * vW[i+1] + (1 - beta2) * np.power(dW[i+1], 2)
                vB[i+1]= beta2 * vB[i+1] + (1 - beta2) * np.power(dB[i+1], 2)

                mW[i+1]= (1.0/(1.0- np.power(beta1 , sample_count))) * mW[i+1]
                mB[i+1]= (1.0/(1.0- np.power(beta1 , sample_count))) * mB[i+1]
                vW[i+1]= (1.0/(1.0- np.power(beta2 , sample_count))) * vW[i+1] 
                vB[i+1]= (1.0/(1.0- np.power(beta2 , sample_count))) * vB[i+1]

                self.W[i+1]-= (lr/np.sqrt(vW[i+1] + epsi)) * mW[i+1]
                self.B[i+1]-= (lr/np.sqrt(vB[i+1] + epsi)) * mB[i+1]

        # Nadam Gradient Descent
        elif opt_algo=='nadam':

          sample_count=0
          for x, y in zip(x_train, y_train):
            self.grad(x, y)
            sample_count+=1
            for i in range(self.h+1): 
              dW[i+1]+= self.dW[i+1] + weight_decay * self.W[i+1]
              dB[i+1]+= self.dB[i+1]

            if sample_count % batch_size == 0:
              mW, mB= {}, {}
              for i in range(self.h+1):
                mW[i+1]= np.zeros(dW[i+1].shape)
                mB[i+1]= np.zeros(dB[i+1].shape)
                mW[i+1]= beta1 * mW[i+1] + (1-beta1) * dW[i+1]
                mB[i+1]= beta1 * mB[i+1] + (1-beta1) * dB[i+1]

                vW[i+1]= beta2 * vW[i+1] + (1-beta2) * np.power(dW[i+1], 2)
                vB[i+1]= beta2 * vB[i+1] + (1-beta2) * np.power(dB[i+1], 2)
                
                mW[i+1]= mW[i+1] / (1- np.power(beta1 , sample_count))
                mB[i+1]= mB[i+1] / (1- np.power(beta1 , sample_count))
                vW[i+1]= vW[i+1] / (1- np.power(beta2 , sample_count))
                vB[i+1]= vB[i+1] / (1- np.power(beta2 , sample_count))
                xW, xB= {}, {}
                xW[i+1]= beta1 * mW[i+1] + (1-beta1) * dW[i+1] / (1- np.power(beta1, sample_count))
                xB[i+1]= beta1 * mB[i+1] + (1-beta1) * dB[i+1] / (1- np.power(beta1, sample_count))
                self.W[i+1]-= ((lr/np.sqrt(vW[i+1] + epsi)) * xW[i+1])
                self.B[i+1]-= ((lr/np.sqrt(vB[i+1] + epsi)) * xB[i+1])

        # Calculating Loss and Accuracy
        y_preds=self.predict(x_train)
        y_val_preds=self.predict(x_val)
        if self.loss_func=='mse':
          loss[e]= mean_squared_error(y_train, y_preds)
          val_loss[e]= mean_squared_error(y_val, y_val_preds)
        else:
          loss[e]= log_loss(y_train, y_preds)
          val_loss[e]= log_loss(y_val, y_val_preds)

        accuracy[e]= accuracy_score(np.argmax(y_preds, axis=1), np.argmax(y_train, axis=1))
        val_accuracy[e]= accuracy_score(np.argmax(y_val_preds, axis=1), np.argmax(y_val, axis=1))

        if accuracy[e] > max_accuracy:
          max_accuracy= accuracy[e]
          max_val_accuracy= val_accuracy[e]
          best_W=self.W.copy()
          best_B=self.B.copy()
        
        # wandb.log({ 'mnist_accuracy': accuracy[e]})
        # wandb.log({ 'epoch': e, 'loss': loss[e], 'val_loss': val_loss[e], 'accuracy': accuracy[e], 'val_accuracy': val_accuracy[e]})

      # Plotting Loss
      if display_accuracy:
        print('Train Accuracy : ', max_accuracy)
        print('Validation Accuracy : ', max_val_accuracy)

      # Plotting Loss
      if display_loss:
        plt.plot(np.asarray(list(loss.values())))
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Loss")
        plt.show()
