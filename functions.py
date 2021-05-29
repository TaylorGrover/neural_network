import numpy as np
"""
Contains activation and loss functions
"""

def relu(z, normalize = True):
   if normalize:
      #return np.maximum(z, 0, z) / z.T.shape[0]
      return z * (z > 0) / z.T.shape[0]
   else:
      return z * (z > 0)
def relu_deriv(y):
   return (y > 0) / y.T.shape[0]

def sigmoid(z):
   return 1 / (np.exp(-z) + 1)
# Sigmoid deriv to be used for the softargmax and for the sigmoid
def sigmoid_deriv(y):
   return y * (1 - y)

# The tanh function's derivative in terms of tanh, where y is the value of tanh(z)
def tanh_deriv(y):
   return 1 - y**2

""" This function anticipates either a matrix of values or a vector, shifting the input to the right by 
the maximum value in each row
"""
def softargmax(z):
   if len(z.shape) == 2:
      z = z - np.max(z, 1)[np.newaxis].T
      exponentials = np.exp(z)
      return exponentials / np.sum(exponentials, 1)[np.newaxis].T
   else:
      z = z - np.max(z)
      exponentials = np.exp(z)
      return exponentials / np.sum(exponentials)

# Can take a batch of inputs and outputs and compute the cost 
def cross_entropy(a, y):
   return -np.sum(np.sum(y * np.log(a) + (1 - y) * np.log(1 - a), 0), 0) / len(a)

def cross_entropy_deriv(a, y):
   return (a - y) / (a * (1 - a))

# Log-likelihood cost
def log_likelihood(a, y):
   return -np.sum(y.T.dot(np.log(a))) / len(a)

# The gradient of log-likelihood with respect to the output layer neurons
def log_likelihood_deriv(a, y):
   return -y.T.dot(1 / a)