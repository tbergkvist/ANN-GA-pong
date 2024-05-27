import numpy as np

class ANN:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        # Initialize the network structure
        self.layers = []
        self.biases = []

        # Create the weight arrays and biases for each layer
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            bias_vector = np.random.randn(layer_sizes[i + 1])
            self.layers.append(weight_matrix)
            self.biases.append(bias_vector)

    def sigmoid(self, x):
        return 2 / (1 + np.exp(-x)) - 1

    def forward(self, x):
        # Forward pass through the network
        for weight_matrix, bias_vector in zip(self.layers, self.biases):
            x = self.sigmoid(np.dot(x, weight_matrix) + bias_vector)
        return x

    def predict(self, state, mult=5):
        # State is a list [ball_x, ball_y, paddle_y, opponent_y]
        input_data = np.array(state)
        output = self.forward(input_data)
        # Convert the output to an integer between -1 and 1
        move = np.clip(output[0], -1, 1)
        move *= 5
        return move
