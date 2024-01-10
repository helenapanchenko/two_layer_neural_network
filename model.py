import numpy as np
from tqdm import tqdm


class TwoLayerNeuralNetwork():
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        # Initialize weights and biases for input to hidden layer
        self.weights_between_input_and_hidden_layers = xavier(input_neurons, hidden_neurons)
        self.biases_between_input_and_hidden_layers = xavier(1, hidden_neurons)

        # Initialize weights and biases for hidden to output layer
        self.weights_between_hidden_and_output_layers = xavier(hidden_neurons, output_neurons)
        self.biases_between_hidden_and_output_layers = xavier(1, output_neurons)

        self.hidden_layer_linear_output = None
        self.hidden_layer_result = None
        self.output_layer_linear_output = None
        self.forward_result = None

        self.loss_history = []
        self.accuracy_history = []

    def forward_pass(self, input_data):
        # Linear output and result for the hidden layer
        self.hidden_layer_linear_output = np.dot(input_data, self.weights_between_input_and_hidden_layers) + self.biases_between_input_and_hidden_layers
        self.hidden_layer_result = sigmoid(self.hidden_layer_linear_output)

        # Linear output and result for the output layer
        self.output_layer_linear_output = np.dot(self.hidden_layer_result,
                                                 self.weights_between_hidden_and_output_layers) + self.biases_between_hidden_and_output_layers
        self.forward_result = sigmoid(self.output_layer_linear_output)

        return self.forward_result

    def backpropagation(self, input_data, targets, alpha=0.1):
        # Backward pass
        # Compute the gradient of the loss with respect to the output layer
        delta_output = (mse_derivative(targets, self.forward_result)
                        * sigmoid_derivative(self.output_layer_linear_output))

        # Compute the gradient of the loss with respect to the hidden layer
        delta_hidden = (np.dot(delta_output, self.weights_between_hidden_and_output_layers.T)
                        * sigmoid_derivative(self.hidden_layer_linear_output))

        # Compute the gradients for the weights and biases
        dw_hidden_output = np.dot(self.hidden_layer_result.T, delta_output) / len(targets)
        db_hidden_output = np.mean(delta_output, axis=0)
        dw_input_hidden = np.dot(input_data.T, delta_hidden) / len(targets)
        db_input_hidden = np.mean(delta_hidden, axis=0)

        # Update weights and biases using gradient descent
        self.weights_between_hidden_and_output_layers -= alpha * dw_hidden_output
        self.biases_between_hidden_and_output_layers -= alpha * db_hidden_output
        self.weights_between_input_and_hidden_layers -= alpha * dw_input_hidden
        self.biases_between_input_and_hidden_layers -= alpha * db_input_hidden

    def train(self, x_train, y_train, epochs, batch_size, alpha=0.1):
        for epoch in range(epochs):
            # Train one epoch
            for i in tqdm(range(0, len(y_train), batch_size)):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Forward step
                y_predicted = self.forward_pass(x_batch)

                # Backpropagation
                self.backpropagation(x_batch, y_batch, alpha)

                # Calculate and store MSE loss for this batch
                loss = mse(y_batch, y_predicted)
                self.loss_history.append(loss)

            # Calculate accuracy on the test set
            epoch_accuracy = accuracy(self, x_train, y_train)
            self.accuracy_history.append(epoch_accuracy)

    def predict(self, data):
        # Forward pass to get predicted probabilities
        predicted_probabilities = self.forward_pass(data)

        # Convert predicted probabilities to class indices
        predicted_classes = np.argmax(predicted_probabilities, axis=1)

        return predicted_classes


def xavier(number_of_inputs, number_of_outputs):
    """
    Initialize a weight matrix using the Xavier/Glorot initialization.

    Parameters:
    - number_of_inputs (int): Number of input units.
    - number_of_outputs (int): Number of output units.

    Returns:
    - np.ndarray: Weight matrix initialized using Xavier/Glorot initialization.
    """
    boundary = np.sqrt(6 / (number_of_inputs + number_of_outputs))
    return np.random.uniform(-boundary, boundary, (number_of_inputs, number_of_outputs))


def sigmoid(data):
    """
    Compute the sigmoid function element-wise.

    The sigmoid function, also known as the logistic function, maps any real-valued number to the range [0, 1].

    Parameters:
    - data (np.ndarray): Input data.

    Returns:
    - np.ndarray: Output of the sigmoid function applied element-wise to the input data.
    """
    return 1 / (1 + np.exp(-data))


def mse(real_values, predicted_values):
    """
    Compute the Mean Squared Error (MSE) between true and predicted values.

    The Mean Squared Error is a measure of the average squared difference between corresponding elements of
    the true and predicted values.

    Parameters:
        real_values (np.ndarray): True values.
        predicted_values (np.ndarray): Predicted values.

    Returns:
        float: Mean Squared Error.
    """
    return ((real_values - predicted_values) ** 2).mean()


def mse_derivative(real_values, predicted_values):
    """
    Compute the derivative of Mean Squared Error (MSE) with respect to predicted values.

    Parameters:
        real_values (np.ndarray): True values.
        predicted_values (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Derivative of MSE with respect to predicted values.
    """
    return 2 * (predicted_values - real_values)


def sigmoid_derivative(data):
    """
    Compute the derivative of the sigmoid function.

    Parameters:
        data (np.ndarray): Input data.

    Returns:
        np.ndarray: Derivative of the sigmoid function.
    """
    return sigmoid(data) * (1 - sigmoid(data))


def accuracy(model, predictors, targets):
    """
    Calculate the accuracy of a model on a given dataset.

    Parameters:
        model: The trained model.
        predictors (np.ndarray): Input data.
        targets (np.ndarray): True labels (one-hot encoded).

    Returns:
        float: Accuracy of the model.

    Raises:
        ValueError: If the length of y_true and y_predicted_probabilities arrays are different.
    """
    # Forward step to get predicted probabilities
    predicted_targets_probabilities = model.forward_pass(predictors)

    if len(targets) != len(predicted_targets_probabilities):
        raise ValueError("Input arrays must have the same length.")

    # Convert predicted probabilities to class indices
    predicted_classes = np.argmax(predicted_targets_probabilities, axis=1)
    # Convert one-hot encoded true labels to class indices
    true_labels = np.argmax(targets, axis=1)

    # Calculate accuracy
    correct_predictions = np.sum(predicted_classes == true_labels)
    total_predictions = len(targets)
    return correct_predictions / total_predictions
