# Created by Teo Bergkvist as a final project in the course EXTG15 at Lund University 2024.

import numpy as np
from ann import ANN


def mutate(ann, mutation_rate=0.01):
    """Function to mutate the weights and biases of the ANN.

    Args:
        ann (ann object): The ANN object to mutate.
        mutation_rate (float): How much to mutate.. Defaults to 0.01.

    Returns:
        ann object: Same object but mutated.
    """
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
    """Crossover function that creates a child ann object.

    Args:
        parent1 (ann object): First parent.
        parent2 (ann object): Second parent.

    Returns:
        ann object: Child ann object.
    """
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
    """Function to update the fitness of the players, based on how many games each player won.

    Args:
        players (list): List of all the players.
        winner_matrix (numpy array): A matrix with all combination of wins(1) or losses(0).

    Returns:
        list: The same players list but with updated fitness values and sorted on them.
    """
    for i, row in enumerate(winner_matrix):
        players[i] = (sum(row), players[i][1])
    players.sort(reverse=True, key=lambda x: x[0])  # Sort players by their fitness
    return players


def multiply(players, ratio_survivors=0.5, mutation_rate=0.01):
    """A function to let a part of the population survive and multiply and the rest die.

    Args:
        players (list): List of all the players.
        ratio_survivors (float): The ratio that survives. Defaults to 0.5. Other values have not been tested (May 2024).
        mutation_rate (float): How much the survivors mutate. Defaults to 0.01.

    Returns:
        list: New generation players.
    """
    number_of_players = len(players)
    for i in range(int(number_of_players // (1 / ratio_survivors))):
        parent1 = mutate(players[i][1], mutation_rate)
        parent2 = mutate(players[i + 1][1], mutation_rate)
        child = crossover(parent1, parent2)
        players[int(number_of_players // (1 / ratio_survivors)) + i] = (0, child)

        players[i] = (0, parent1)  # Set parents fitness to 0.
        players[i + 1] = (0, parent2)
    return players
