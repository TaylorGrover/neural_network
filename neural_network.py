import json
import numpy as np
np.set_printoptions(linewidth=100)
import os

# Use fixed random seed FOR TESTING
np.random.seed(1)

def sigmoid(z):
   return 1 / (np.exp(-z) + 1)
# Differential equation x(1 - x)
def deriv(x):
   return x * (1 - x)

# wb_filename is a JSON file containing the weights and biases
class NeuralNetwork:
   def __init__(self, layer_heights = [], weights = [], biases = [], wb_filename = ""):
      self.weights = weights
      self.biases = biases
      self.layer_heights = layer_heights
      if(os.path.isfile(wb_filename)):
         self.get_weights(wb_filename)
      else:
         self.randomize()
      self.wb_filename = wb_filename

   # TODO  Reads the saved weights and biases from a file
   def get_weights(self, filename):
      with open(filename, "r") as f:
         weights, biases = json.load(f)
         layer_heights = [len(weights[i]) for i in range(len(weights))]

   def randomize(self):
      w = []
      b = []
      for i in range(len(self.layer_heights) - 1):
         w.append(2 * np.random.random((self.layer_heights[i], self.layer_heights[i + 1])) - 1)
         b.append(2 * np.random.random(self.layer_heights[i + 1]) - 1)
      self.weights = w
      self.biases = b

   # Forward feed the input data into the network. x is the input layer
   def feed(self, x, training = False):
      current_layer = np.array(x)
      if training:
         pass
      else:
         for i in range(len(self.layer_heights) - 1):
            print(current_layer)
            current_layer = self.f(np.dot(current_layer, self.weights[i]) + self.biases[i])
      return current_layer

   # Override f with a lambda expression
   def f(self, x):
      return x
   
   # The derivative of f
   def fp(self, x):
      return x
# this code alone is worth MILLIONS
test_network = NeuralNetwork([3, 4, 2])
test_network.f = sigmoid
test_network.fp = deriv
# MATHHH
v = np.array([1, 2, 3])

print(test_network.feed(v))