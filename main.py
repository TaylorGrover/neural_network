from functions import *
import numpy as np
import sys
np.set_printoptions(linewidth=200)
from extract import *
from neural_network import NeuralNetwork

### Hyperparameters
epochs = 20
batch_size = 10
eta = .5
decay = .0000000007
architecture = [784, 30, 10]
activations = ["sigmoid"]

# Use fixed random seed FOR TESTING
#np.random.seed(1)

# test weights
wb_test_filename = "test_wb.json"

# Image training data files
directory = "data/"
training_image_filename = "train-images-idx3-ubyte"
training_labels_filename = "train-labels-idx1-ubyte"
testing_image_filename = "t10k-images-idx3-ubyte"
testing_labels_filename = "t10k-labels-idx1-ubyte"

# Create a neural network to read the image inputs from 
nn = NeuralNetwork(architecture, activations)
nn.cost = cross_entropy
nn.cost_deriv = cross_entropy_deriv

# Get the image data for both training and validation
training_images, training_labels, validation_images, validation_labels \
   = get_training_and_validation(directory + training_image_filename, directory + training_labels_filename)

# Get the testing image data and labels
testing_images = get_images_array(directory + testing_image_filename)
testing_labels = get_labels(directory + testing_labels_filename)

if len(sys.argv) == 3:
   wb_filename = sys.argv[1]
   nn.set_wb(wb_filename)
   if sys.argv[2] == "test":
      print(nn.test([testing_images, testing_labels], iterations = "all"))

# Training the network with starting weights
if len(sys.argv) == 1 or sys.argv[2] == "train":
   nn.train([training_images, training_labels], batch_size = batch_size, epochs = epochs, eta = eta, show_stats = True)

# hello
# 1
# 2 
# 3
# 4
# 5, last neuron