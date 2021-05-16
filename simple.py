"""
Test very simple neural network with very simple data.
"""

from neural_network import *

# H.P. (hyperparameters)
epochs = 10000
architecture = [4, 4, 2]
eta = .1
batch_size = 3
activations = ["sigmoid"]

inputs = np.array([[1, 0, 0, 1],
                   [0, 1, 0, 1],
                   [0, 0, 1, 0]])
outputs = np.array([[0, 0], 
                    [1, 0],
                    [0, 1]])

simpleNetwork = NeuralNetwork(layer_heights = architecture, activations = activations)

simpleNetwork.train([inputs, outputs], epochs=epochs, batch_size=1, eta=eta, show_stats=True)