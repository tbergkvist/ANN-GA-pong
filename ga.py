# Created by Teo Bergkvist as a final project in the course EXTG15 at Lund University 2024.

import numpy as np
from ann import ANN


def mutate(ann, mutation_rate=0.01):
    for i in range(len(ann.layers)):
        mutation_mask_weights = np.random.rand(*ann.layers[i].shape) < mutation_rate
        ann.layers[i][mutation_mask_weights] += np.random.normal(
            scale=0.1, size=ann.layers[i].shape
        )[mutation_mask_weights]

        mutation_mask_biases = np.random.rand(*ann.biases[i].shape) < mutation_rate
        ann.biases[i][mutation_mask_biases] += np.random.normal(
            scale=0.1, size=ann.biases[i].shape
        )[mutation_mask_biases]

    return ann


def crossover(parent1, parent2):
    child_layers = []
    child_biases = []

    for layer1, layer2 in zip(parent1.layers, parent2.layers):
        mask = np.random.rand(*layer1.shape) > 0.5
        child_layer = np.where(mask, layer1, layer2)
        child_layers.append(child_layer)

    for bias1, bias2 in zip(parent1.biases, parent2.biases):
        mask = np.random.rand(*bias1.shape) > 0.5
        child_bias = np.where(mask, bias1, bias2)
        child_biases.append(child_bias)

    child = ANN(
        input_size=len(parent1.layers[0]),
        hidden_layer_sizes=[layer.shape[1] for layer in parent1.layers[:-1]],
        output_size=parent1.layers[-1].shape[1],
    )
    child.layers = child_layers
    child.biases = child_biases

    return child


def update_fitness(players, winner_matrix):
    for i, row in enumerate(winner_matrix):
        players[i] = (sum(row), players[i][1])
    players.sort(reverse=True, key=lambda x: x[0])  # Sort players by their fitness
    return players


def multiply(players, ratio_survivors=0.5, mutation_rate=0.01):
    number_of_players = len(players)
    for i in range(int(number_of_players // (1 / ratio_survivors))):
        parent1 = mutate(players[i][1], mutation_rate)
        parent2 = mutate(players[i + 1][1], mutation_rate)
        child = crossover(parent1, parent2)
        players[int(number_of_players // (1 / ratio_survivors)) + i] = (0, child)

        players[i] = (0, parent1)  # Set parents fitness to 0.
        players[i + 1] = (0, parent2)
    return players
