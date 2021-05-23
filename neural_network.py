from functions import *
import json
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

# wb_filename is a JSON file containing the weights and biases
class NeuralNetwork:
   def __init__(self, architecture = [], activations = [], wb_filename = ""):
      self.architecture = architecture
      self.layer_count = len(self.architecture)
      self.wb_filename = wb_filename
      if(os.path.isfile(self.wb_filename)):
         self.set_wb(self.wb_filename)
      else:
         self.randomize()
      self.layers = [[] for i in range(self.layer_count)]
      self._get_activations(activations)

   # Reads the saved weights and biases from a file
   def set_wb(self, filename):
      with open(filename, "r") as f:
         weights, biases = json.load(f)
      self.architecture = [len(weights[i]) for i in range(len(weights))]
      self.architecture.append(len(biases[-1]))
      self.layer_count = len(self.architecture)
      self.weights = weights
      self.biases = biases
      for i in range(self.layer_count - 1):
         self.weights[i] = np.array(self.weights[i])
         self.biases[i] = np.array(self.biases[i])

   # gives the weight and bias a value, if no wb_filename provided
   def randomize(self):
      w = []
      b = []
      for i in range(self.layer_count - 1):
         #w.append(2 * np.random.random((self.architecture[i], self.architecture[i + 1])) - 1)
         #b.append(2 * np.random.random(self.architecture[i + 1]) - 1)
         w.append(np.random.randn(self.architecture[i], self.architecture[i + 1]) / np.sqrt(self.architecture[0]))
         #b.append(np.random.randn(self.architecture[i + 1]))
         b.append(np.zeros(self.architecture[i + 1]))
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

   """
   data has the format (training_images, training_labels)
   epochs is 1 by default
   batch_size: number of images and labels per training iteration
   eta: the learning rate; can be adjusted over time
   save_wb: whether the weights and biases should be preserved in a json file
   show_stats: save the cost into an array, show the cost of each batch, then display a plot of the cost over time (number of batches)
   """
   def train(self, data, epochs = 1, batch_size = 10, eta = 1, save_wb = True, show_stats = False): 
      self.delta_w, self.delta_b = self._init_deltas()
      self.correct = 0
      eta_0 = eta
      current_batch = 0
      self.costs = [0 for i in range(epochs * int(len(data[0]) / batch_size))]
      for i in range(epochs):
         print("Epoch: %d" % (i + 1))
         #eta = eta_0 * (.5) ** i
         batches = self._get_batches(data, batch_size)
         for item_input, desired_output in zip(*batches):
            self.delta_w, self.delta_b = self.backprop(item_input, desired_output)
            self._update_wb(eta, len(item_input))
            current_batch += 1
            if eta == 0:
               break
            if show_stats:
               self.costs[current_batch - 1] = self.cost(self.layers[-1], desired_output)
               self.print_output(desired_output, eta, current_batch - 1)

      # Save weights and biases after training. If show_stats, then display a graph of the cost over time
      if save_wb:
         filename = self._generate_filename()
         self.save_wb(filename)
         if show_stats:
            self.plot_cost(filename)

   def plot_cost(self, filename):
      plt.ion()
      fig, ax = plt.subplots()
      ax.plot(np.arange(0, len(self.costs), 1), self.costs)
      fig.savefig(filename.replace("json", "png"))

   """
   Can be overridden. Mean-squared error (quadratic cost) by default. 
   Takes actual output and desired output then returns the cost
   """
   def cost(self, a, y):
      err = .5 * np.linalg.norm(np.sum(a - y, 0)) ** 2 / len(a)
      return err

   """
   The derivative (gradient) of the cost with respect to the activated output layer neurons. 
   Takes desired and actual outputs as params. 
   """
   def cost_deriv(self, a, y):
      return a - y

   def print_output(self, desired_outputs, eta, index):
      batch_size = len(desired_outputs)
      diff = np.sum(np.abs(np.round(self.layers[-1]) - desired_outputs), 1)
      incorrect = np.sum(diff * (diff > 0))
      self.correct += batch_size - incorrect
      print("Cost/Batch: %.10f\tEta: %.10f\tAccuracy: %.10f" % (self.costs[index], eta, self.correct / (batch_size * (index + 1))))


   """ 
   inputs is a batch matrix containing input row vectors. Backprop feeds the inputs matrix to the network, preserving the layers, then computes
   the gradient of the weights and biases. When computing the derivative of the activation functions with respect to the z term (weighted sum + bias),
   save the weighted sums in addition to the activated layers (will have to choose one or the other later)
   """
   def backprop(self, x, y):
      dw, db = self._init_deltas()
      self.layers[0] = x
      for i in range(self.layer_count - 1):
         self.layers[i + 1] = self.f[i](np.dot(self.layers[i], self.weights[i]) + self.biases[i])

      delta_l = self.cost_deriv(self.layers[-1], y) * self.fp[-1](self.layers[-1])
      for i in range(self.layer_count - 2, -1, -1):
         dw[i] = np.sum((self.layers[i][np.newaxis].T * delta_l).swapaxes(0, 1), 0)
         #dw[i] = sum(np.array([np.array([self.layers[i][j]]).T * delta_l[j] for j in range(len(delta_l))]))
         db[i] = np.sum(delta_l, 0)
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
