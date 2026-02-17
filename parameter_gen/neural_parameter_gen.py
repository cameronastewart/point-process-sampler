import numpy as np

# set random seed (first five digits of the golden ratio)
np.random.seed(16180)

dim = 100

weightsInit = np.random.normal(scale=(1 / dim) ** 0.5, size=(dim, dim))
weights = np.tril(weightsInit) + np.tril(weightsInit, -1).T
biases = 5 * np.ones(dim)

np.savez("neural_parameters", weights=weights, biases=biases)