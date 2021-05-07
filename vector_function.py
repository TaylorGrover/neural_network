import numpy as np
np.set_printoptions(linewidth = 200)

def sigmoid(x):
   return 1 / (np.exp(-x) + 1)
