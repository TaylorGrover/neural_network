import json
import numpy as np
import os

# wb_filename is a JSON file containing the weights and biases
class NeuralNetwork:
   def __init__(self, layer_heights = [], wb_filename = ""):
      self.layer_heights = layer_heights
      self.layer_count = len(self.layer_heights)
      self.wb_filename = wb_filename
      if(os.path.isfile(self.wb_filename)):
         self.get_wb(self.wb_filename)
      else:
         self.randomize()
      self.layers = [[] for i in range(self.layer_count)]

   # TODO  Reads the saved weights and biases from a file
   def get_wb(self, filename):
      with open(filename, "r") as f:
         weights, biases = json.load(f)
         layer_heights = [len(weights[i]) for i in range(len(weights))]
         layer_heights.append(biases[-1])
         self.layer_heights = layer_heights
         self.layer_count = len(self.layer_heights)
         self.weights = weights
         self.biases = biases

   # gives the weight and bias a value, if none provided
   def randomize(self):
      w = []
      b = []
      for i in range(self.layer_count - 1):
         w.append(2 * np.random.random((self.layer_heights[i], self.layer_heights[i + 1])) - 1)
         b.append(2 * np.random.random(self.layer_heights[i + 1]) - 1)
      self.weights = w
      self.biases = b

   # Forward feed the input data into the network. x is the input layer
   def feed(self, x, training = False, eta = .01):
      current_layer = np.array(x)
      # BIG FUNCTIONALITY BIG GRADIENT DESCENT BIG BACKPROPAGATION
      if training:
         self.layers[0] = current_layer
         for i in range(self.layer_count - 1):
            current_layer = self.f(np.dot(current_layer, self.weights[i]) + self.biases[i])
            self.layers[i + 1] = current_layer
      else:
         for i in range(self.layer_count - 1):
            current_layer = self.f(np.dot(current_layer, self.weights[i]) + self.biases[i])
      return current_layer

   def train(self, data, batch_size = 1, eta = .01):
      inputs, outputs = data
      print(inputs[0])
      print(outputs[0])


   # Override f with a lambda expression
   def f(self, x):
      return x
   
   # The derivative of f. Should be overridden with an actual derivative to the 
   def fp(self, x):
      return x

   # Save the weights and the biases after the network is trained or whenever we want
   def save_wb(self, param_file):
      wb = [[], []]
      for w in self.weights:
         wb[0].append(w.tolist())
      for b in self.biases:
         wb[1].append(b.tolist())
      with open(param_file, "w") as f:
         json.dump(wb, f)
