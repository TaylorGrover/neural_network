import pdb
from functions import *
import json
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
#np.seterr(all="raise")
import os
import sys
import time

"""
Initialize the network with the number of neurons in each layer (architecture). 
TODO: Activation function derivatives as differential equations in terms of the output only, or written explicitly in terms of z.
   In the first case, it's only necessary to preserve the activated layers to compute the gradient, but the latter requires the 
   weighted sum (inactivated)
TODO: Organize boolean switches (use_diff_eq, save_wb, show_cost, test_validation, dropout, L1, L2)
"""
class NeuralNetwork:
   def __init__(self, architecture = [], activations = [], wb_filename = "", cost = "MSE", use_clipping = False, use_dropout = False, use_L2 = False, show_gradient = False, save_wb = False, show_cost = True):
      self.architecture = architecture
      self.activations = activations
      self.layer_count = len(self.architecture)
      self.wb_filename = wb_filename
      if cost == "MSE":
         self.cost = MSE
         self.cost_deriv = MSE_deriv
      elif cost == "cross_entropy":
         self.cost = cross_entropy
         self.cost_deriv = cross_entropy_deriv
      self.use_clipping = use_clipping
      self.use_dropout = use_dropout
      self.use_L2 = use_L2
      self.show_gradient = show_gradient
      self.save_wb = save_wb
      self.show_cost = show_cost
      if(os.path.isfile(self.wb_filename)):
         self.set_wb(self.wb_filename)
      else:
         self.randomize()
         self._get_activation_functions(activations)
      if len(self.activations) != len(self.architecture) - 1:
         raise ValueError("Need to have %d activation functions for network with %d layers." % (self.layer_count - 1, self.layer_count))
      self.layers = [[] for i in range(self.layer_count)]
      self.zs = [[] for i in range(self.layer_count)]

   # Reads the saved weights and biases from a file
   def set_wb(self, filename):
      with open(filename, "r") as f:
         weights, biases = json.load(f)
      self.architecture = [len(weights[i]) for i in range(len(weights))]
      self.architecture.append(len(biases[-1]))
      self.layer_count = len(self.architecture)
      self._get_activation_functions(filename.split("_"), is_file = True)
      self.biases = biases
      self.weights = weights
      for i in range(self.layer_count - 1):
         self.weights[i] = np.array(self.weights[i])
         self.biases[i] = np.array(self.biases[i])

   """
   Initialize the weights and biases
   """
   def randomize(self):
      w = []
      b = []
      for i in range(self.layer_count - 1):
         #w.append(2 * np.random.random((self.architecture[i], self.architecture[i + 1])) - 1)
         #b.append(2 * np.random.random(self.architecture[i + 1]) - 1)
         w.append(np.random.randn(self.architecture[i], self.architecture[i + 1]) / np.sqrt(self.architecture[0]))
         b.append(np.random.randn(self.architecture[i + 1]) + 1)
         #b.append(np.zeros(self.architecture[i + 1]))
      self.weights = w
      self.biases = b

   def set_diff_eq(self, boolean):
      self.use_diff_eq = boolean
      self._get_activation_functions(self.activations)

   def feed(self, x):
      """
      x is a vector (array/list)
      """
      current_layer = np.array(x)
      for i in range(self.layer_count - 1):
         current_layer = self.f[i](np.dot(current_layer, self.weights[i]) + self.biases[i])
      return current_layer

   def test(self, X_test, y_test, iterations = "all", interval = 20, show_class_accuracies=True):
      """
      data is a tuple (inputs, outputs) 
      The inputs are fed to the network which gives some result to be compared with the outputs.
      The output components of the network are rounded to 0 or 1 then tested for equality with the
       corresponding desired output. When testing the network, this will output the desired vs 
       actual output every interval.
      """
      total_correct = 0
      if iterations == "all":
         iterations = len(X_test)

      # calculate the accuracy
      estimated = np.round(self.feed(X_test))
      column_sum = np.sum(estimated * y_test, axis=0)
      accuracy = np.sum(column_sum) / iterations

      if show_class_accuracies:
         # Show the individual accuracies for each class
         class_counts = np.sum(y_test, axis=0)
         print(column_sum / class_counts)

      return accuracy

   """
   activations is a list of strings which contain information about the network activation functions, and possibly the neuronal architecture.
   If the is_file boolean is true, this skips the first n strings in the list and uses the next n - 1 strings for the activations. If you're 
   using the activation layer to compute the derivative with a differential 
   """
   def _get_activation_functions(self, activations, is_file = False):
      self.f = []
      self.fp = []
      if is_file:
         self.activations = activations[self.layer_count : 2 * self.layer_count - 1]
      else:
         self.activations = activations
      a_dict = activations_diff_dict
      for activation in self.activations:
         self.f.append(a_dict[activation][0])
         self.fp.append(a_dict[activation][1])
         

   """
   data has the format (train_data, train_label, validation_data, validation_label, test_data, test_label)
   epochs is 1 by default
   batch_size: number of inputs and labels per training iteration
   eta: the learning rate; can be adjusted over time
   save_wb: whether the weights and biases should be preserved in a json file
   show_cost: save the cost into an array, show the cost of each batch, then display a plot of the cost over time (number of batches)
   decay: L2 regularization decay rate
   clip_threshold: threshold value to use for gradient clipping
   After training is complete, if save_wb is true, obtain testing data classification accuracy, then save the weights and biases.
   """
   def train(self, X_train, y_train, X_test, y_test, validation=None, epochs = 1, batch_size = 10, eta = 1, decay = 0, clip_threshold = 0): 
      if self.use_clipping and clip_threshold <= 0:
         raise ValueError("Clip threshold should be greater than 0.")

      # Intialize the backprop deltas
      self.delta_w, self.delta_b = self._init_deltas()
      self.correct = 0

      # Initialize the validation data
      self.validations = []
      eta_0 = eta
      current_batch = 0

         # Get the test data
      self.costs = [0 for i in range(epochs * int(len(X_train) / batch_size))]
      for i in range(epochs):
         print("Epoch: %d" % (i + 1))
         #eta = eta_0 * (.5) ** i
         if hasattr(validation, "__len__") and len(validation) == 2:
            self.validations.append(self.test(list(validation)))
         batches = self._get_batches([X_train, y_train], batch_size)
         for item_input, desired_output in zip(*batches):
            self.delta_w, self.delta_b = self.backprop(item_input, desired_output)
            self._update_wb(eta, decay, clip_threshold)
            current_batch += 1
            if self.show_cost:
               self.costs[current_batch - 1] = self.cost(self.layers[-1], desired_output)
               #eta = 1 / self.costs[current_batch - 1] 
               self.print_output(desired_output, eta, current_batch - 1)
      # Save weights and biases after training. If show_cost, then display a graph of the cost over time
      if self.save_wb:
         accuracy = self.test([X_test, y_test])
         filename = self._generate_filename(accuracy)
         self.save_model(filename)
         if self.show_cost:
            self.plot_cost(filename)

   def backprop(self, x, y):
      """ 
      x is a batch matrix containing input row vectors. Backprop feeds the inputs 
      matrix to the network, preserving the layers, then computes the gradient of 
      the weights and biases. 
      """
      dw, db = self._init_deltas()
      self.layers[0] = x
      self.zs[0] = x
      for i in range(self.layer_count - 1):
         self.zs[i + 1] = np.dot(self.layers[i], self.weights[i]) + self.biases[i]
         self.layers[i + 1] = self.f[i](self.zs[i + 1])
         # If we're using dropout and we're not at the last layer
         if self.use_dropout and i + 1 < self.layer_count - 1:
            self.layers[i + 1] *= np.random.randint(0, (2,) * self.layers[i + 1].T.shape[0])
         #print(self.layers[i + 1])
      #delta_l = self.cost_deriv(self.layers[-1], y) * self.fp[-1](self.layers[-1])
      delta_l = self._get_output_delta(y)
      for i in range(self.layer_count - 2, -1, -1):
         dw[i] = np.average((self.layers[i][np.newaxis].T * delta_l).swapaxes(0, 1), 0)
         db[i] = np.average(delta_l, 0)
         delta_l = np.dot(delta_l, self.weights[i].T) * self.fp[i](self.layers[i])
      return dw, db 

   def _get_output_delta(self, y):
      """
      Return the delta associated with the last layer's weights
      """
      if self.cost == cross_entropy and (self.f[-1] == sigmoid or self.f[-1] == softargmax):
         return self.layers[-1] - y
      return self.cost_deriv(self.layers[-1], y) * self.fp[-1](self.layers[-1])

   def _update_wb(self, eta, decay, clip_threshold):
      """
      Update weights and biases with self.delta_w and self.delta_b at the end 
      of each batch
      """
      for i in range(self.layer_count - 1):
         #print("i: %d\tMax: %.2f\tMin: %.2f\tAvg: %.2f\tMedian: %.2f\tStddev: %.2f\tShape: %s" % (i, np.max(self.delta_w[i]), np.min(self.delta_w[i]), np.average(self.delta_w[i]), np.median(self.delta_w[i]), np.std(self.delta_w[i]), repr(self.delta_w[i].shape)))
         #if np.max(self.delta_w[i]) < 3 and np.abs(np.min(self.delta_w[i])) < 3:
         if self.use_clipping:
            norm = np.linalg.norm(self.delta_w[i])
            if norm >= clip_threshold or norm <= 1e-7:
               self.delta_w[i] /= (norm / clip_threshold)
         self.weights[i] = (1 - eta * decay) * self.weights[i] - eta * self.delta_w[i]
         if self.show_gradient:
            print("Layer: %d\tAvg: %.5f\tMax: %.5f\tMin: %.5f\tStddev: %.5f" % (i + 1, np.average(self.delta_w[i]), np.max(self.delta_w[i]), np.min(self.delta_w[i]), np.std(self.delta_w[i])))
         self.biases[i] -= eta * self.delta_b[i]

   def plot_cost(self, filename):
      """
      Display and save a plot of the cost of the training data over the total 
      number of batches. The plots are saved in a directory called "plots" and 
      have the same naming convention and the weights and biases files.
      """
      plt.ion()
      fig, ax = plt.subplots()
      ax.plot(np.arange(0, len(self.costs), 1), self.costs)
      if not os.path.isdir("plots"):
         os.mkdir("plots")
      fig.savefig("plots/" + filename.replace("json", "png"))

   def cost(self, a, y):
      """
      Can be overridden. Mean-squared error (quadratic cost) by default. 
      Takes actual output and desired output then returns the cost
      """
      err = .5 * np.linalg.norm(np.sum(a - y, 0)) ** 2 / len(a)
      return err

   def cost_deriv(self, a, y):
      """
      The derivative (gradient) of the cost with respect to the activated output 
      layer neurons. Takes desired and actual outputs as params. 
      """
      return a - y

   def print_output(self, desired_outputs, eta, index):
      """
      Print the cost of each batch during training and the values of the learning 
      rate, and show the accumulated accuracy of training data classification.
      """
      #batch_size = len(desired_outputs)
      #diff = np.sum(np.abs(np.round(self.layers[-1]) - desired_outputs), 1)
      #incorrect = np.sum(diff > 0)
      #self.correct += batch_size - incorrect
      if index % 20 == 0:
         print("Batch number: %d\tCost/Batch: %.10f\t\U0001d702: %.10f\tCMean: %.10f\tCSTDDEV: %.10f" % (index, self.costs[index], eta, np.mean(self.costs[0 : index + 1]), np.std(self.costs[0 : index + 1]))) #self.correct / (batch_size * (index + 1))))

   def _init_deltas(self):
      """
      The deltas are the gradients with respect to each layer of weights and 
      biases, and they have the same shape as the weights and biases arrays. 
      They are first initialized to zero, but are later set to the values of 
      the gradient. 
      """
      return [np.zeros(w.shape) for w in self.weights], [np.zeros(b.shape) for b in self.biases]
   
   def _get_batches(self, data, batch_size):
      """
      data is a tuple; data[0] contains all the input vectors, data[1] the 
        corresponding labels. 
      zipped is a sequence of tuples of the inputs with their labels. 
      batch_size is the chosen length of each input batch
      """
      zipped = list(zip(*data))
      np.random.shuffle(zipped)
      inputs, outputs = zip(*zipped)
      n = len(zipped)
      input_batches = [np.array(inputs[i : i + batch_size]) for i in range(0, n, batch_size)]
      output_batches = [np.array(outputs[i : i + batch_size]) for i in range(0, n, batch_size)]
      return input_batches, output_batches

   """
   The weights and biases filename is generated based on the architecture, the activation
   functions, and the percent accuracy on the testing data.
   """
   def _generate_filename(self, accuracy):
      name = ""
      for size in self.architecture:
         name += str(size) + "_"
      for activation in self.activations:
         name += activation + "_"
      name += "%.1f" % (np.round(accuracy, 3) * 100) + ".json"
      return name

   """
   Put the weights and biases into an array, then save them to a json file.
   """
   def save_model(self, param_file):
      wb = [[], []]
      for w in self.weights:
         wb[0].append(w.tolist())
      for b in self.biases:
         wb[1].append(b.tolist())
      if not os.path.isdir("wb"):
         os.mkdir("wb")
      with open("wb/" + param_file, "w") as f:
         json.dump(wb, f)
