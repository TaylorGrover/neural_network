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
         self.set_wb(self.wb_filename)
      else:
         self.randomize()
      self.layers = [[] for i in range(self.layer_count)]

   # Reads the saved weights and biases from a file
   def set_wb(self, filename):
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

   """
   x is a vector (array/list)
   if training, then the network saves the activation layers
   """
   def feed(self, x, training = False):
      current_layer = np.array(x)
      if training:
         self.layers[0] = current_layer
         for i in range(self.layer_count - 1):
            current_layer = self.f(np.dot(current_layer, self.weights[i]) + self.biases[i])
            self.layers[i + 1] = current_layer
      else:
         for i in range(self.layer_count - 1):
            current_layer = self.f(np.dot(current_layer, self.weights[i]) + self.biases[i])
         return current_layer

   def test(self, data, cost_threshold, iterations):
      total_correct = 0
      inputs, outputs = data
      for i, (x, y) in enumerate(zip(inputs, outputs)):
         if i >= iterations:
            break
         if self.cost(x, y) <= cost_threshold:
            total_correct += 1
      return total_correct / len(data)


   def f(self, z):
      return 1 / (np.exp(-z) + 1)
   
   """
   fp should be written as a differential equation in terms of the output of f
   """
   def fp(self, y):
      return y * (1 - y)

   # x, y = data, where x is training inputs and y is the set of corresponding desired outputs. 
   # datum
   def train(self, data, epochs = 1, batch_size = 48, eta = .001, show_cost_deriv = False):
      self.delta_w, self.delta_b = self._init_deltas()
      for i in range(epochs):
         print("Epoch: %d" % i)
         batches = self._get_batches(data, batch_size)
         for batch in batches:
            for item in batch:
               dw, db = self.backprop(item)
               self.delta_w = [w + wn for w, wn in zip(self.delta_w, dw)]
               self.delta_b = [b + bn for b, bn in zip(self.delta_b, db)]
            self._update_wb(eta, len(batch))
            if show_cost_deriv:
               self.print_output(item)
      # Save weights and biases after training
      self.save_wb(self._generate_filename())

   def cost(self, x, y):
      # For debugging
      a = self.feed(x)
      print("Actual output: " + repr(np.round(a)))
      print("Desired output: " + repr(y) + "\n")
      err = .5 * np.linalg.norm(a - y) ** 2
      print("Cost of single input: " + str(err))
      return err

   """
   The derivative (gradient) of the cost with respect to the activated output layer neurons
   """
   def cost_deriv(self, x, y):
      return self.feed(x) - y

   def print_output(self, item):
      print("\nActual: " + repr(self.layers[-1]))
      print("Desired: " + repr(item[1]) + "\n")
      
   """ 
   For each weight matrix and bias vector in self.weights and self.biases,
   compute the gradient of C with respect to the current weights and biases with
   the given input and output in item. Item is a tuple of the input vector and 
   the desired output vector.
   """
   def backprop(self, item):
      x, y = item
      dw, db = self._init_deltas()
      self.feed(x, training = True)
      delta_l = (self.layers[-1] - y) * self.fp(self.layers[-1])
      for i in range(self.layer_count - 2, -1, -1):
         dw[i] = np.array([self.layers[i]]).T * delta_l
         db[i] = delta_l
         delta_l = np.dot(delta_l, self.weights[i].T) * self.fp(self.layers[i])
      return dw, db 

   """
   Update weights and biases with self.delta_w and self.delta_b at the end of each batch
   """

   def _update_wb(self, eta, batch_size):
      for i in range(self.layer_count - 1):
         self.weights[i] -= eta * self.delta_w[i] / batch_size
         self.biases[i] -= eta * self.delta_b[i] / batch_size

   def _init_deltas(self):
      return [np.zeros(w.shape) for w in self.weights], [np.zeros(b.shape) for b in self.biases]
   
   """
   data is a tuple; data[0] contains all the image vectors, data[1] the corresponding labels.
   zipped is a sequence of tuples of the images with their labels.
   batches is the list of minibatches as numpy arrays
   """
   def _get_batches(self, data, batch_size):
      zipped = list(zip(*data))
      np.random.shuffle(zipped)
      n = len(zipped)
      batches = [zipped[i : i + batch_size] for i in range(0, n, batch_size)]
      return batches


   def _generate_filename(self):
      time_struct = time.localtime()
      name = ""
      for i in range(6):
         name += str(time_struct[i]) + "_"
      name += "wb.json"
      return name

   # Save the weights and the biases after the network is trained or whenever we want
   def save_wb(self, param_file):
      wb = [[], []]
      for w in self.weights:
         wb[0].append(w.tolist())
      for b in self.biases:
         wb[1].append(b.tolist())
      with open(param_file, "w") as f:
         json.dump(wb, f)
