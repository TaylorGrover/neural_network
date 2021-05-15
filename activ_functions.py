import numpy as np
"""
Contains most of the activation functions and their derivatives
"""

def relu(z):
   return z * (z > 0)
def relu_deriv(y):
   return y > 0

def sigmoid(z):
   return 1 / (np.exp(-z) + 1)
# Sigmoid deriv to be used for the softargmax and for the sigmoid
def sigmoid_deriv(y):
   return y * (1 - y)

def softargmax(z):
   exponentials = np.exp(z)
   return exponentials / sum(exponentials)