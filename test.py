from neural_network import *

architecture = [3, 200, 200, 2]
activations = ["relu", "relu", "relu"]
batch_size = 3
epochs = 2000
eta = .01
decay = .1

# Neural network with quadratic cost
nn = NeuralNetwork(architecture, activations)
nn.save_wb = False
nn.show_gradient = True
nn.use_dropout = True
nn.use_L2 = True

# Inputs
x = np.random.randn(3, 3)

# Desired outputs
y = np.array([[3, 5],
              [7, 11],
              [13, 17]])

nn.train([x, y, x, y, x, y], epochs = epochs, batch_size = batch_size, eta = eta, decay = decay)