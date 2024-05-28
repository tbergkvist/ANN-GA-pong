# Created by Teo Bergkvist as a final project in the course EXTG15 at Lund University 2024.

import numpy as np
import json


class ANN:
    def __init__(self, input_size, hidden_layer_sizes, output_size, gpu=False):
        """Initializes a neural network object.

        Args:
            input_size (int): Number of input parameters to the network.
            hidden_layer_sizes (list): A list where each item is the number of neurons in that layer.
            output_size (int): Number of output parameters from the network.
        """
        self.layers = []
        self.biases = []
        self.random_number = np.random.random()
        self.gpu = gpu
        self.cp = None
        self.predict_multiplier = 5

        # Create the weight arrays and biases for each layer.
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            bias_vector = np.random.randn(layer_sizes[i + 1])
            self.layers.append(weight_matrix)
            self.biases.append(bias_vector)

        if self.gpu:
            import cupy as cp
            self.cp = cp

    @staticmethod
    def tanh(x):
        """A tanh function. Pushes the x values to be in the range -1, 1.

        Args:
            x (int): Value to put into the function.

        Returns:
            int: Function output.
        """
        return np.tanh(x / 2)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (array): Input values.

        Returns:
            int: Output value.
        """
        for weight_matrix, bias_vector in zip(self.layers, self.biases):
            x = np.dot(x, weight_matrix) + bias_vector
            x = self.tanh(x)
        return x
    
    def forward_gpu(self, x):
        """Forward pass through the network using GPU acceleration with cupy.

        Args:
            x (array): Input values.

        Returns:
            int: Output value.
        """
        x = self.cp.array(x)  # Convert input to cupy array
        for weight_matrix, bias_vector in zip(self.layers, self.biases):
            weight_matrix_cp = self.cp.array(weight_matrix)  # Convert weights to cupy array
            bias_vector_cp = self.cp.array(bias_vector)  # Convert biases to cupy array
            x = self.cp.dot(x, weight_matrix_cp) + bias_vector_cp
            x = self.tanh(x)
        return self.cp.asnumpy(x)  # Convert result back to numpy array

    def predict(self, state):
        """Use the forward method to process state values and output a direction value.

        Args:
            state (array): Input values.

        Returns:
            int: Direction values.
        """
        # State is a list [ball_x, ball_y, paddle_y, opponent_y].
        input_data = np.array(state)
        if self.gpu:
            output = self.forward_gpu(input_data)
        else:
            output = self.forward(input_data)
        # Convert the output to an integer between -1 and 1.
        move = np.clip(output[0], -1, 1)
        move *= self.predict_multiplier
        return move

    def __lt__(self, other):
        """Comparasion function that is needed when fitness values are the same.

        Args:
            other (float): From the other neural net object.

        Returns:
            bool: Comparasion result.
        """
        return self.random_number < other.random_number

    def save(self, filename):
        """Save the ANN weights and biases to a file.

        Args:
            filename (string): Filename to save to.
        """
        data = {
            "layers": [weight_matrix.tolist() for weight_matrix in self.layers],
            "biases": [bias_vector.tolist() for bias_vector in self.biases],
        }
        with open(filename, "w") as file:
            json.dump(data, file)

    def load(self, filename):
        """Load weights and biases from a file and update the ANN object.

        Args:
            filename (string): Filename to read from.
        """
        with open(filename, "r") as file:
            data = json.load(file)
            self.layers = [np.array(weight_matrix) for weight_matrix in data["layers"]]
            self.biases = [np.array(bias_vector) for bias_vector in data["biases"]]
