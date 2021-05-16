from activ_functions import *
import json
import numpy as np
import os
import sys
import time

# wb_filename is a JSON file containing the weights and biases
class NeuralNetwork:
   def __init__(self, layer_heights = [], activations = [], wb_filename = ""):
      self.layer_heights = layer_heights
      self.layer_count = len(self.layer_heights)
      self.wb_filename = wb_filename
      if(os.path.isfile(self.wb_filename)):
         self.set_wb(self.wb_filename)
      else:
         self.randomize()
      self.layers = [[] for i in range(self.layer_count)]
      self.zlayers = [[] for i in range(self.layer_count)]
      self._get_activations(activations)

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
   """
   def feed(self, x):
      current_layer = np.array(x)
      for i in range(self.layer_count - 1):
         current_layer = self.f[i](np.dot(current_layer, self.weights[i]) + self.biases[i])
      return current_layer

   def test(self, data, iterations = "all"):
      total_correct = 0
      inputs, outputs = data
      if iterations == "all":
         iterations = len(inputs)
      for i, (x, y) in enumerate(zip(inputs, outputs)):
         if i >= iterations:
            break
         classification = np.round(self.feed(x))
         if np.array_equal(classification, y):
            total_correct += 1
         if i % 20 == 0:
            print("Act: " + repr(classification) + "\nDes: " + repr(y), end="\n\n")
      return total_correct / i

   def _get_activations(self, activations):
      self.f = []
      self.fp = []
      if "relu" in activations:
         for i in range(self.layer_count - 2):
            self.f.append(relu)
            self.fp.append(relu_deriv)
      else:
         for i in range(self.layer_count - 2):
            self.f.append(sigmoid)
            self.fp.append(sigmoid_deriv)
      if "softargmax" in activations:
         self.f.append(softargmax)
         self.fp.append(sigmoid_deriv)
      else:
         self.f.append(sigmoid)
         self.fp.append(sigmoid_deriv)

   # x, y = data, where x is training inputs and y is the set of corresponding desired outputs. 
   def train(self, data, epochs = 1, batch_size = 48, eta = 1, show_stats = False, decay = .001):
      self.delta_w, self.delta_b = self._init_deltas()
      total_batches = 0
      for i in range(epochs):
         print("Epoch: %d" % (i + 1))
         batches = self._get_batches(data, batch_size)
         for item_input, desired_output in zip(*batches):
            self.delta_w, self.delta_b = self.backprop(item_input, desired_output)
            self._update_wb(eta, len(item_input))
            total_batches += 1
            eta *= (1 - decay * sigmoid(total_batches))
            if eta == 0:
               break
            if show_stats:
               self.print_output(desired_output, eta)

      # Save weights and biases after training
      self.save_wb(self._generate_filename())

   """
   Can be overridden. Mean-squared error (quadratic cost) by default
   """
   def cost(self, x, y):
      # For debugging
      a = self.feed(x)
      err = .5 * np.linalg.norm(a - y) ** 2
      return err

   """
   The derivative (gradient) of the cost with respect to the activated output layer neurons.
   """
   def cost_deriv(self, x, y):
      return self.feed(x) - y

   def print_output(self, desired_outputs, eta):
      batch_size = len(desired_outputs)
      print("Act: " + repr(np.round(self.layers[-1][-1])) + "\nDes: " + repr(desired_outputs[-1]))
      print("Cost/Batch: %.3f\tEta: %.3f" % (1 / batch_size * np.linalg.norm(self.layers[-1] - desired_outputs) ** 2 / 2, eta)) 


   """ 
   inputs is a batch matrix containing input row vectors. Backprop feeds the inputs matrix to the network, preserving the layers, then computes
   the gradient of the weights and biases. When computing the derivative of the activation functions with respect to the z term (weighted sum + bias),
   save the weighted sums in addition to the activated layers (will have to choose one or the other later)
   """
   def backprop(self, x, y):
      dw, db = self._init_deltas()
      self.layers[0] = x
      self.zlayers[0] = x
      for i in range(self.layer_count - 1):
         self.zlayers[i + 1] = np.dot(self.layers[i], self.weights[i]) + self.biases[i]
         self.layers[i + 1] = self.f[i](self.zlayers[i + 1])

      delta_l = (self.layers[-1] - y) * self.fp[-1](self.layers[-1])
      for i in range(self.layer_count - 2, -1, -1):
         dw[i] = sum((np.array([self.layers[i]]).T * delta_l).swapaxes(0, 1))
         #dw[i] = sum(np.array([np.array([self.layers[i][j]]).T * delta_l[j] for j in range(len(delta_l))]))
         db[i] = sum(delta_l)
         delta_l = np.dot(delta_l, self.weights[i].T) * self.fp[i](self.layers[i])
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
      inputs, outputs = zip(*zipped)
      n = len(zipped)
      input_batches = [np.array(inputs[i : i + batch_size]) for i in range(0, n, batch_size)]
      output_batches = [np.array(outputs[i : i + batch_size]) for i in range(0, n, batch_size)]
      return input_batches, output_batches

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
