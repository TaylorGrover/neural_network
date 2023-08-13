import numpy as np

"""
Activation and cost functions. The activation function derivatives in terms of z end in "deriv" and end in "diff" when in terms of the activation output.
"""

def identity(z):
   return z
def identity_deriv(z):
   return np.ones(z.shape)
def identity_diff(y):
   return np.ones(y.shape)

def lrelu(z):
   if z > 0:
      return z
   return -0.01 * z
def lrelu_deriv(z):
   if z > 0:
      return np.ones(z.shape)
   return -0.01

def normalized_relu(z):
   return z * (z > 0) / (z.T.shape[0])
def normalized_relu_deriv(z):
   return (z > 0) / z.T.shape[0]
def normalized_relu_diff(y):
   return (y > 0) / (y.T.shape[0])

def relu(z):
   return z * (z > 0)
def relu_deriv(z):
   return z > 0
def relu_diff(y):
   return y > 0

def sigmoid(z):
   return 1 / (np.exp(-z) + 1)
def sigmoid_deriv(z):
   s_z = sigmoid(z)
   return s_z * (1 - s_z)
# Sigmoid diff to be used for the softargmax and for the sigmoid
def sigmoid_diff(y):
   return y * (1 - y)

# The tanh function's derivative in terms of tanh, where y is the value of tanh(z)
def tanh_deriv(z):
   return 1 / np.cosh(z) ** 2
def tanh_diff(y):
   return 1 - y**2

def sin_squared(z):
   return np.sin(z) ** 2
def sin_squared_deriv(z):
   return np.sin(2 * z)

""" This function anticipates either a matrix of values or a row vector, shifting the input to the right by 
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

def softargmax_deriv(z):
   activation = softargmax(z)
   return activation * (1 - activation)

def softplus(z):
   return np.log(1 + np.exp(z))
def softplus_diff(y):
   return 1 - np.exp(-y)

# Can take a batch of inputs and outputs and compute the cost 
def cross_entropy(a, y):
   return -np.sum(np.sum(y * np.log(a) + (1 - y) * np.log(1 - a), 0), 0) / len(a)

def cross_entropy_deriv(a, y):
   return (a - y) / (a * (1 - a))

# MSE
def MSE(a, y):
    return .5 * np.linalg.norm(np.sum(a - y, 0)) ** 2 / len(a)

def MSE_deriv(a, y):
    return a - y

# Log-likelihood cost
def log_likelihood(a, y):
   return -np.sum(y.T.dot(np.log(a))) / len(a)

# The gradient of log-likelihood with respect to the output layer neurons
def log_likelihood_deriv(a, y):
   return -y.T.dot(1 / a)

"""
Dictionary of activation functions
"""
activations_diff_dict = {"relu" : (relu, relu_diff), "nrelu" : (normalized_relu, normalized_relu_diff), \
   "sigmoid" : (sigmoid, sigmoid_diff), "tanh" : (np.tanh, tanh_diff), \
   "softargmax" : (softargmax, sigmoid_diff), "identity" : (identity, identity_diff), "softplus" : (softplus, softplus_diff)}

activations_dict = {"relu" : (relu, relu_deriv), "lrelu" : (lrelu, lrelu_deriv), "nrelu" : (normalized_relu, normalized_relu_deriv), \
   "sigmoid" : (sigmoid, sigmoid_deriv), "tanh" : (np.tanh, tanh_deriv), "sinsquared" : (sin_squared, sin_squared_deriv), \
   "softargmax" : (softargmax, softargmax_deriv)}
