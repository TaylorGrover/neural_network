from neural_network import *
from functions import *
from extract import *
"""
Select a trained digit classifier to check the accuracy of the generator.
"""

# Hyperparameters
epochs = 1
batch_size = 10
eta = .3
decay = 5.0
architecture = [10, 80, 784]
activations = ["sigmoid", "sigmoid"]

# Generator
generator = NeuralNetwork(architecture, activations)

# Training images and labels (the labels are the training inputs and the images are the desired outputs in this case)
