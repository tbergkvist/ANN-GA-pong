import numpy as np
from concurrent.futures import ThreadPoolExecutor
from simulation import *
from ann import ANN
from ga import *


def run_single_game(game, row, col):
    """Runs a single game and returns the result in a tuple.

    Args:
        game: The game to run.
        row: The row index of the game in the game_matrix.
        col: The column index of the game in the game_matrix.

    Returns:
        tuple: The row, col, and winner of the game.
    """
    if game is not None:
        winner = run_headless(game)
        if winner.name == "Player 1":
            return (row, col, 1, 0)
        elif winner.name == "Player 2":
            return (row, col, 0, 1)
    return (row, col, None, None)


def run_games(game_matrix):
    """Runs all the games in the game_matrix.

    Args:
        game_matrix (numpy array): The matrix containing the games to run.

    Returns:
        numpy array: A matrix that contains the results of the games.
    """
    winner_matrix = np.zeros((len(game_matrix), len(game_matrix)))
    tasks = []
    with ThreadPoolExecutor() as executor:
        for row in range(len(game_matrix)):
            for col in range(len(game_matrix)):
                if row != col:
                    game = game_matrix[row][col]
                    tasks.append(executor.submit(run_single_game, game, row, col))
        for task in tasks:
            row, col, result_row, result_col = task.result()
            if result_row is not None:
                winner_matrix[row][col] = result_row
                winner_matrix[col][row] = result_col
    return winner_matrix


gpu = False
input_size = 4  # ball_x, ball_y, paddle_y, opponent_y, do not change this if you do not know what you are doing!!
hidden_layer_sizes = [5, 5]  # Two hidden layers with 10 neurons each.
output_size = 1  # Single output neuron for the paddle movement.
number_of_players = 50  # The number of players to have in each generation.
number_of_generations = 10  # The number of generations to run.
mutation_rate = 0.01
players = []

for i in range(number_of_players):  # Create all players.
    players.append((0, ANN(input_size, hidden_layer_sizes, output_size, gpu=gpu)))  # Tuples with their score and their ANN.

for i in range(number_of_generations):
    print(f"Simulating generation: {i} out of {number_of_generations}. {round(i / number_of_generations, 4) * 100}%")
    game_matrix = create_game_matrix(players)
    winner_matrix = run_games(game_matrix)
    update_fitness(players, winner_matrix)
    players = multiply(players, mutation_rate=mutation_rate)

number_of_players_to_save = 4  # This sets the number of players to save to a file.

for i in range(number_of_players_to_save):
    players[i][1].save(f"./ANN_player_{i}.json")
