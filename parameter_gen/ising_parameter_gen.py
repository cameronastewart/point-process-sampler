import numpy as np

# set random seed (first five digits of e)
np.random.seed(27182)

dim = 100

weightsInit = np.random.normal(scale=(4 / dim) ** 0.5, size=(dim, dim))
weights = np.tril(weightsInit, -1) + np.tril(weightsInit, -1).T
biases = weights.sum(axis=1)

np.savez("ising_parameters", weights=weights, biases=biases)