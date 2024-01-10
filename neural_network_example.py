import os

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt

from model import TwoLayerNeuralNetwork
from sklearn.metrics import accuracy_score


def one_hot(data: np.ndarray) -> np.ndarray:
    """
    Convert a 1-dimensional array of categorical labels into a one-hot encoded matrix.

    Parameters:
    - data (np.ndarray): A 1-dimensional NumPy array containing categorical labels.

    Returns:
    - np.ndarray: A 2-dimensional NumPy array representing the one-hot encoded matrix.
      Each row corresponds to an element in the input array, and each column
      represents a unique label with 1 indicating the presence of the label and 0 otherwise.
    """
    train_targets = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    train_targets[rows, data] = 1
    return train_targets


def plot(loss_history: list, accuracy_history: list, filename='plot'):
    """
    Visualize the learning process by plotting loss and accuracy over epochs.

    Parameters:
    - loss_history (list): A list containing loss values for each epoch.
    - accuracy_history (list): A list containing accuracy values for each epoch.
    - filename (str, optional): The name of the file to save the plot. Default is 'plot'.

    Returns:
    - None: This function does not return anything; it saves the plot as an image file.
    """
    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


def scale(x_train, x_test):
    """
    Scale the input data by dividing each feature by its maximum value.

    Parameters:
    - x_train (np.ndarray): Training data features.
    - x_test (np.ndarray): Test data features.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Scaled training and test data.
    """
    max_values = x_train.max(axis=0)
    max_values[max_values == 0] = 1
    return x_train / max_values, x_test / max_values


if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train_dataset = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test_dataset = pd.read_csv('../Data/fashion-mnist_test.csv')
    train_predictors = raw_train_dataset[raw_train_dataset.columns[1:]].values
    test_predictors = raw_test_dataset[raw_test_dataset.columns[1:]].values

    encoded_train_labels = one_hot(raw_train_dataset['label'].values)
    test_labels = raw_test_dataset['label']

    scaled_train_predictors, scaled_test_predictors = scale(train_predictors, test_predictors)

    model_instance = TwoLayerNeuralNetwork(scaled_train_predictors.shape[1], 64, 10)
    model_instance.train(scaled_train_predictors, encoded_train_labels, 50, 100, 0.25)
    print([round(acc, 2) for acc in model_instance.accuracy_history])
    predicted_labels = model_instance.predict(scaled_test_predictors)
    print(accuracy_score(test_labels, predicted_labels))
