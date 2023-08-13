import numpy as np
import sys
np.set_printoptions(linewidth=200)
from extract import *
from neural_network import NeuralNetwork

### Hyperparameters
epochs = 3
batch_size = 10
clip_threshold = .01
eta = .3
decay = 0
architecture = [784, 20, 10]
activations = ["sigmoid", "softargmax"]

# Use fixed random seed FOR TESTING
np.random.seed(1)

""" Create a neural network to read the image inputs from. Choose the appropriate cost function 
   and cost gradient (with respect to the output layer). MSE is the default cost function
"""
nn = NeuralNetwork(architecture, activations, cost="cross_entropy")

# Get the image data for both training and validation
training_images, training_labels, validation_images, validation_labels = get_training_and_validation()

# Get the testing image data and labels
testing_images, testing_labels = get_testing_images()

if len(sys.argv) == 3:
   wb_filename = sys.argv[1]
   nn.set_wb(wb_filename)
   if sys.argv[2] == "test":
      print(nn.test(testing_images, testing_labels, iterations = "all"))

# Training the network with starting weights
if len(sys.argv) == 1 or sys.argv[2] == "train":
   nn.train(training_images, training_labels, testing_images, testing_labels, validation=[validation_images, validation_labels], batch_size = batch_size, clip_threshold = clip_threshold, decay = decay, epochs = epochs, eta = eta)

