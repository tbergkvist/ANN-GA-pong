# Created by Teo Bergkvist as a final project in the course EXTG15 at Lund University 2024.
import numpy as np
import json


class ANN:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        # Initialize the network structure
        self.layers = []
        self.biases = []
        self.random_number = np.random.random()

        # Create the weight arrays and biases for each layer
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            bias_vector = np.random.randn(layer_sizes[i + 1])
            self.layers.append(weight_matrix)
            self.biases.append(bias_vector)

    @staticmethod
    def sigmoid(x):
        return 2 / (1 + np.exp(-x)) - 1

    def forward(self, x):
        # Forward pass through the network using efficient NumPy operations
        for weight_matrix, bias_vector in zip(self.layers, self.biases):
            x = np.dot(x, weight_matrix) + bias_vector
            x = self.sigmoid(x)  # Apply sigmoid in a vectorized manner
        return x

    def predict(self, state, mult=5):
        # State is a list [ball_x, ball_y, paddle_y, opponent_y]
        input_data = np.array(state)
        output = self.forward(input_data)
        # Convert the output to an integer between -1 and 1
        move = np.clip(output[0], -1, 1)
        move *= mult
        return move

    def __lt__(self, other):
        return self.random_number < other.random_number

    def save(self, filename):
        # Convert weights and biases to lists and save to a file
        data = {
            "layers": [weight_matrix.tolist() for weight_matrix in self.layers],
            "biases": [bias_vector.tolist() for bias_vector in self.biases],
        }
        with open(filename, "w") as file:
            json.dump(data, file)

    def load(self, filename):
        # Load weights and biases from a file and convert them back to NumPy arrays
        with open(filename, "r") as file:
            data = json.load(file)
            self.layers = [np.array(weight_matrix) for weight_matrix in data["layers"]]
            self.biases = [np.array(bias_vector) for bias_vector in data["biases"]]
