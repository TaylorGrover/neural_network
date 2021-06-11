from functions import *
import numpy as np
import sys
np.set_printoptions(linewidth=200)
from extract import *
from neural_network import NeuralNetwork

### Hyperparameters
epochs = 20
batch_size = 10
eta = .3
decay = 1
architecture = [784, 10, 10]
activations = ["sigmoid", "softargmax"]

# Use fixed random seed FOR TESTING
np.random.seed(1)

""" Create a neural network to read the image inputs from. Choose the appropriate cost function 
   and cost gradient (with respect to the output layer). MSE is the default cost implementation
"""
nn = NeuralNetwork(architecture, activations)
nn.cost = cross_entropy
nn.cost_deriv = cross_entropy_deriv
nn.test_validation = False
nn.save_wb = False
#nn.use_diff_eq = False

# Get the image data for both training and validation
training_images, training_labels, validation_images, validation_labels = get_training_and_validation()

# Get the testing image data and labels
testing_images, testing_labels = get_testing_images()

if len(sys.argv) == 3:
   wb_filename = sys.argv[1]
   nn.set_wb(wb_filename)
   if sys.argv[2] == "test":
      print(nn.test([testing_images, testing_labels], iterations = "all"))

# Training the network with starting weights
if len(sys.argv) == 1 or sys.argv[2] == "train":
   nn.train([training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels], \
       batch_size = batch_size, epochs = epochs, eta = eta, decay = decay)