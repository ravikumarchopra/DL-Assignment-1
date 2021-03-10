import numpy as np

class FFNNetwork:

    def __init__(self, input_size, output_size=1, hidden_layers=[2]):
      self.x=input_size
      self.y=output_size
      self.h=len(hidden_layers)
      self.sizes=[self.x] + hidden_layers + [self.y]
      self.W={}
      self.B={}

      for i in range(self.h+1):
        self.W[i+1]=np.random.randn(self.sizes[i], self.sizes[i+1])
        self.B[i+1]=np.zeros((1, self.sizes[i+1]))

    def sigmoid(self, x):
      return (1.0/(1.0 + np.exp(-x)))

    def softmax(self, y):
      exps=np.exp(y)
      return np.exp(y)/np.sum(exps)

    def forward_pass(self, x):
      self.A={}
      self.H={}
      self.H[0]= x
      
      for i in range(self.h+1):
        self.A[i+1]=np.matmul(self.H[i], self.W[i+1]) + self.B[i+1]
        self.H[i+1]=self.sigmoid(self.A[i+1])
      self.H[self.h+1]=self.softmax(self.A[self.h+1])
      return self.H[self.h+1]