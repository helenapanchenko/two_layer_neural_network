# Two-Layer Neural Network from Scratch
This project demonstrates the fundamental architecture and mechanics behind a fully-connected two-layer Neural Network (NN) built from scratch, without using any high-level machine learning libraries such as TensorFlow or PyTorch. This implementation is intended for educational purposes, providing a clear understanding of the NN's forward and backward passes.

## Theoretical Background
A two-layer neural network consists of an input layer, one hidden layer, and an output layer. Each neuron in a layer is connected to all neurons in the subsequent layer, hence the term 'fully-connected'.

### Forward Pass
In the forward pass, the network takes the input data and passes it through the hidden layer, then through the output layer to make a prediction. The transformation at each layer is a combination of a linear transformation (achieved through weights and biases) and a non-linear transformation (using an activation function like sigmoid).

### Backward Pass (Backpropagation)
After the forward pass, the network calculates the loss to determine the error of its prediction. During the backpropagation, the network adjusts its weights and biases to minimize this error. It computes the gradient of the loss function with respect to the weights and applies gradient descent to update the parameters.

### Mean Squared Error (MSE) and Derivatives
MSE is used as the loss function to quantify the error between the predicted and actual outputs. During backpropagation, we use its derivative (gradient) to update the network's weights.

## Usage
To use the model:

1. Initialize the model by specifying the size of the input, hidden, and output layers.
2. Train the model with training data using the 'train' method.
3. Make predictions on new data using the 'predict' method.

Customizable hyperparameters include:

* Number of neurons in the hidden layer
* Learning rate ('alpha')
* Number of epochs for training
* Batch size for mini-batch gradient descent 

## Customization
You can customize the model according to your needs. Here are a few things you might consider changing:

* __Number of Hidden Layers__: While the current model has one hidden layer, you can easily extend it to multiple hidden layers.
* __Activation Functions__: Replace the sigmoid function with other activation functions like ReLU or Tanh for the hidden layers.
* __Loss Function__: The model uses MSE, but you can implement and use cross-entropy or other loss functions.
* __Learning Rate (Alpha)__: Adjust the learning rate to control the speed of learning.
* __Epochs__: Set the number of training cycles over the entire dataset.
* __Batch Size__: For larger datasets, you can implement batch training to process the data in chunks.

## Contributing
Feel free to fork this project and submit pull requests with improvements or open an issue if you find any bugs or have suggestions.