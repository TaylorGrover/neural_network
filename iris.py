"""
Test the neural network on the iris data set
"""
from neural_network import NeuralNetwork
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.set_printoptions(linewidth=200)

# Get the iris dataset
iris = pd.read_csv("data/iris.data", header=None)

# Function to map iris labels
def encode_labels(t):
    if t == "Iris-setosa": return np.array([1, 0, 0])
    elif t == "Iris-versicolor": return np.array([0, 1, 0])
    else: return np.array([0, 0, 1])

X = np.array(iris[[0, 1, 2, 3]])
y = np.array(list(iris[4].map(encode_labels)))

# Standard scale the data
X = StandardScaler().fit_transform(X)

# Perform a train/test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Set NN hyperparameters
batch_size = 10
epochs = 100
architecture = [X.shape[1], 300, y.shape[1]]
activations = ["sigmoid", "softargmax"]

nn = NeuralNetwork(architecture, activations, cost="cross_entropy")
nn.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)
print(nn.test(X_test, y_test))
print(np.round(nn.feed(X_test)) * y_test)
