from neural_network import *

class GAN(NeuralNetwork):
    def __init__(self, architecture = [], activations = [], wb_filename = ""):
        super().__init__(architecture, activations, wb_filename)