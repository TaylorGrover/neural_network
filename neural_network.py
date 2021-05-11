import json
import numpy as np
import os
import time

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

   # gives the weight and bias a value, if no wb_filename provided
   def randomize(self):
      w = []
      b = []
      for i in range(self.layer_count - 1):
         w.append(2 * np.random.random((self.layer_heights[i], self.layer_heights[i + 1])) - 1)
         b.append(2 * np.random.random(self.layer_heights[i + 1]) - 1)
      self.weights = w
      self.biases = b

   # Forward feed the input data into the network. x is the input layer
   def feed(self, x, training = False):
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

   # x, y = data, where x is training inputs and y is the set of corresponding desired outputs. 
   # datum
   def train(self, data, epochs = 1, batch_size = 1, eta = .01):
      batches = self._get_batches(data, batch_size)
      print(len(batches[0]))
      for i in range(epochs):
         for batch in batches:
            delta_L = 0
            for datum in batch:
               cur_input, des_output = datum
               delta_L += self.feed(cur_input, training = True) - des_output
            delta_L /= len(batch)
            delta_L *= self.fp(self.layers[-1])
            print(delta_L)
      # Save weights and biases after training
      self.save_wb(self._generate_filename())
      
   
   def _get_batches(self, data, batch_size):
      inputs, desired_outputs = data
      full_batch = []
      length = len(inputs)
      for i in range(length):
         full_batch.append((inputs[i], desired_outputs[i]))
      np.random.shuffle(full_batch)
      batches = []
      for i in range(length):
         if i % batch_size == 0:
            batches.append([])
         batches[i // batch_size].append(full_batch[i])
      return batches

   def _generate_filename(self):
      time_struct = time.localtime()
      name = ""
      for i in range(6):
         name += str(time_struct[i]) + "_"
      name += "wb.json"
      return name

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
