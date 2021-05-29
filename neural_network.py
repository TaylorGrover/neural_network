from functions import *
import json
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

"""
Either pass both architecture and activations (i.e. "relu", "softargmax", "sigmoid"), or just wb_filename, since set_wb initializes 
the architecture and activation functions from the filename string.
"""
class NeuralNetwork:
   def __init__(self, architecture = [], activations = [], wb_filename = ""):
      self.architecture = architecture
      self.layer_count = len(self.architecture)
      self.wb_filename = wb_filename
      if(os.path.isfile(self.wb_filename)):
         self.set_wb(self.wb_filename)
      else:
         self.randomize()
         self._get_activation_functions(activations)
      self.layers = [[] for i in range(self.layer_count)]

   # Reads the saved weights and biases from a file
   def set_wb(self, filename):
      with open(filename, "r") as f:
         weights, biases = json.load(f)
      self.architecture = [len(weights[i]) for i in range(len(weights))]
      self.architecture.append(len(biases[-1]))
      self.layer_count = len(self.architecture)
      self._get_activation_functions(filename.split("_"))
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

   """
   x is a vector (array/list)
   """
   def feed(self, x):
      current_layer = np.array(x)
      for i in range(self.layer_count - 1):
         current_layer = self.f[i](np.dot(current_layer, self.weights[i]) + self.biases[i])
      return current_layer

   """
   data is a tuple (inputs, outputs) 
   The inputs are fed to the network which gives some result to be compared with the outputs.
   The output components of the network are rounded to 0 or 1 then tested for equality with the
    corresponding desired output. When testing the network, this will output the desired vs 
    actual output every interval.
   """
   def test(self, data, iterations = "all", interval = 20, show_outputs=True):
      total_correct = 0
      inputs, outputs = data
      if iterations == "all":
         iterations = len(inputs)
      for i, (x, y) in enumerate(zip(inputs, outputs)):
         if i >= iterations:
            break
         current_output = self.feed(x)
         if np.array_equal(np.round(current_output), y):
            total_correct += 1
         if i % interval == 0 and show_outputs:
            print("Act: " + repr(current_output) + "\nDes: " + repr(y), end="\n\n")
      accuracy = total_correct / (i + 1)
      print("Accuracy: " + str(accuracy))
      return accuracy

   """
   activations is an array of strings of the names of activation functions. The names
   can be "relu", "softargmax", "sigmoid", or "tanh". If "relu" is passed, then the "relu" activation function
   will be used for all layers preceding the output layer, then sigmoid applied to the output
   layer. If "sigmoid" is the only string present, sigmoid will be used for all layers. If 
   "softargmax" and "relu" are in activations, then "relu" will be in the hidden layers and softargmax
   will be used in the output layer.
   """
   def _get_activation_functions(self, activations):
      self.f = []
      self.fp = []
      self.activations = []
      if "relu" in activations:
         self.activations.append("relu")
         hidden_layer_activation_function = relu
         hidden_layer_activation_deriv = relu_deriv
      elif "tanh" in activations:
         self.activations.append("tanh")
         hidden_layer_activation_function = np.tanh
         hidden_layer_activation_deriv = tanh_deriv
      else:
         self.activations.append("sigmoid")
         hidden_layer_activation_function = sigmoid
         hidden_layer_activation_deriv = sigmoid_deriv
      for i in range(self.layer_count - 2):
         self.f.append(hidden_layer_activation_function)
         self.fp.append(hidden_layer_activation_deriv)
      if "softargmax" in activations or "softmax" in activations:
         self.activations.append("softargmax")
         self.f.append(softargmax)
         self.fp.append(sigmoid_deriv)
      else:
         if "sigmoid" not in self.activations:
            self.activations.append("sigmoid")
         self.f.append(sigmoid)
         self.fp.append(sigmoid_deriv)

   """
   data has the format (train_data, train_label, validation_data, validation_label, test_data, test_label)
   epochs is 1 by default
   batch_size: number of inputs and labels per training iteration
   eta: the learning rate; can be adjusted over time
   save_wb: whether the weights and biases should be preserved in a json file
   show_cost: save the cost into an array, show the cost of each batch, then display a plot of the cost over time (number of batches)
   After training is complete, if save_wb is true, obtain testing data classification accuracy, then save the weights and biases.
   """
   def train(self, data, epochs = 1, batch_size = 10, eta = 1, decay = .1, save_wb = True, show_cost = True, test_validation=True): 
      self.delta_w, self.delta_b = self._init_deltas()
      self.correct = 0
      validation_accuracy = []
      eta_0 = eta
      current_batch = 0
      td, tl, vd, vl, tstd, tstl = data
      self.costs = [0 for i in range(epochs * int(len(td) / batch_size))]
      for i in range(epochs):
         print("Epoch: %d" % (i + 1))
         #eta = eta_0 * (.5) ** i
         if test_validation:
            self.test([vd, vl])
         batches = self._get_batches([td, tl], batch_size)
         for item_input, desired_output in zip(*batches):
            self.delta_w, self.delta_b = self.backprop(item_input, desired_output)
            self._update_wb(eta, len(item_input), decay)
            current_batch += 1
            if eta == 0:
               break
            if show_cost:
               self.costs[current_batch - 1] = self.cost(self.layers[-1], desired_output)
               self.print_output(desired_output, eta, current_batch - 1)
      # Save weights and biases after training. If show_cost, then display a graph of the cost over time
      if save_wb:
         accuracy = self.test([tstd, tstl])
         filename = self._generate_filename(accuracy)
         self.save_wb(filename)
         if show_cost:
            self.plot_cost(filename)

   """
   Display and save a plot of the cost of the training data over the total number of batches.
   The plots are saved in a directory called "plots" and have the same naming convention and 
   the weights and biases files.
   """
   def plot_cost(self, filename):
      plt.ion()
      fig, ax = plt.subplots()
      ax.plot(np.arange(0, len(self.costs), 1), self.costs)
      if not os.path.isdir("plots"):
         os.mkdir("plots")
      fig.savefig("plots/" + filename.replace("json", "png"))

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

   """
   Print the cost of each batch during training and the values of the learning rate, and show the
   accumulated accuracy of training data classification.
   """
   def print_output(self, desired_outputs, eta, index):
      #batch_size = len(desired_outputs)
      #diff = np.sum(np.abs(np.round(self.layers[-1]) - desired_outputs), 1)
      #incorrect = np.sum(diff > 0)
      #self.correct += batch_size - incorrect
      if index % 20 == 0:
         print("Batch number: %d\tCost/Batch: %.10f\tEta: %.10f" % (index, self.costs[index], eta)) #self.correct / (batch_size * (index + 1))))


   """ 
   inputs is a batch matrix containing input row vectors. Backprop feeds the inputs matrix to the network, 
   preserving the layers, then computes the gradient of the weights and biases. 
   """
   def backprop(self, x, y):
      dw, db = self._init_deltas()
      self.layers[0] = x
      for i in range(self.layer_count - 1):
         self.layers[i + 1] = self.f[i](np.dot(self.layers[i], self.weights[i]) + self.biases[i])
      #delta_l = self.cost_deriv(self.layers[-1], y) * self.fp[-1](self.layers[-1])
      delta_l = self._get_output_delta(y)
      for i in range(self.layer_count - 2, -1, -1):
         dw[i] = np.sum((self.layers[i][np.newaxis].T * delta_l).swapaxes(0, 1), 0)
         db[i] = np.sum(delta_l, 0)
         delta_l = np.dot(delta_l, self.weights[i].T) * self.fp[i](self.layers[i])
      return dw, db 

   def _get_output_delta(self, y):
      if self.cost == cross_entropy and (self.f[-1] == sigmoid or self.f[-1] == softargmax):
         return self.layers[-1] - y
      return self.cost_deriv(self.layers[-1], y) * self.fp[-1](self.layers[-1])

   """
   Update weights and biases with self.delta_w and self.delta_b at the end of each batch
   """
   def _update_wb(self, eta, batch_size, decay):
      for i in range(self.layer_count - 1):
         self.weights[i] = (1 - eta * decay / batch_size) * self.weights[i] - eta * self.delta_w[i] / batch_size
         self.biases[i] -= eta * self.delta_b[i] / batch_size

   """
   The deltas are the gradients with respect to each layer of weights and biases, and they 
   have the same shape as the weights and biases arrays. They are first initialized to zero,
   but are later set to the values of the gradient. 
   """
   def _init_deltas(self):
      return [np.zeros(w.shape) for w in self.weights], [np.zeros(b.shape) for b in self.biases]
   
   """
   data is a tuple; data[0] contains all the input vectors, data[1] the corresponding labels.
   zipped is a sequence of tuples of the inputs with their labels.
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
   def save_wb(self, param_file):
      wb = [[], []]
      for w in self.weights:
         wb[0].append(w.tolist())
      for b in self.biases:
         wb[1].append(b.tolist())
      if not os.path.isdir("wb"):
         os.mkdir("wb")
      with open("wb/" + param_file, "w") as f:
         json.dump(wb, f)
