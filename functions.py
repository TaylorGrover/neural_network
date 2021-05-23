import numpy as np
"""
Contains activation and loss functions
"""

def relu(z):
   return z * (z > 0)
def relu_deriv(y):
   if y == 0:
      return .5
   return y > 0

def sigmoid(z):
   return 1 / (np.exp(-z) + 1)
# Sigmoid deriv to be used for the softargmax and for the sigmoid
def sigmoid_deriv(y):
   return y * (1 - y)

# 
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