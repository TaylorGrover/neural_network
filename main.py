import numpy as np
np.set_printoptions(linewidth=200)
from extract import *
from neural_network import NeuralNetwork

# Use fixed random seed FOR TESTING
np.random.seed(1)

def sigmoid(z):
   return 1 / (np.exp(-z) + 1)

# Differential equation x' = x(1 - x)
def deriv(x):
   return x * (1 - x)


# test weights
wb_test_filename = "test_wb.json"

# Image training data files
directory = "data/"
training_image_filename = "train-images-idx3-ubyte"
training_labels_filename = "train-labels-idx1-ubyte"

# GET THE IMAGE DATA AND LABELS
training_images = get_images_array(directory + training_image_filename)
training_labels = get_labels(directory + training_labels_filename)

# Create a neural network to read the image inputs from 
test_network = NeuralNetwork([784, 15, 10])
test_network.f = sigmoid
test_network.fp = deriv

test_network.train([training_images, training_labels])

# 1
# 2 
# 3
# 4
# 5, last neuron