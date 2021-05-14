import numpy as np
import sys
np.set_printoptions(linewidth=200)
from extract import *
from neural_network import NeuralNetwork

### Hyperparameters
epochs = 10
batch_size = 48
eta = 1
architecture = [784, 80, 50, 10]

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
test_network = NeuralNetwork(architecture)

if(len(sys.argv) == 1):

   # GET THE IMAGE DATA AND LABELS
   training_images = get_images_array(directory + training_image_filename)
   training_labels = get_labels(directory + training_labels_filename)

   # Training the network
   test_network.train([training_images, training_labels], batch_size = batch_size, epochs = epochs, eta = eta, show_stats = True)
if(len(sys.argv) == 2):
   # GET THE TESTING IMAGE DATA AND LABELS
   wb_test_filename = sys.argv[1]
   testing_images = get_images_array(directory + testing_image_filename)
   testing_labels = get_labels(directory + testing_labels_filename)

   # Set the weights and biases from trained wb json file
   test_network.set_wb(wb_test_filename)

   # Test the data
   print(test_network.test([testing_images, testing_labels], batch_size = 1, cost_threshold = .3, iterations = 50))

# hello
# 1
# 2 
# 3
# 4
# 5, last neuron