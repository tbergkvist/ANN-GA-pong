import numpy as np

def mutate(weights, mutation_rate=0.01):
    for key in weights.keys():
        mutation_mask = (np.random.rand(*weights[key].shape) < mutation_rate)
        weights[key][mutation_mask] += np.random.normal(scale=0.1, size=weights[key].shape)[mutation_mask]
    return weights

def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        child[key] = parent1[key].copy()
        mask = np.random.rand(*child[key].shape) > 0.5
        child[key][mask] = parent2[key][mask]
    return child