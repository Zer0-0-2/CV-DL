# What is a Neural Network?

A Nural Network is a algebraic function in its core. theres multiple stages that make up the "learning" process of it. It works by manipulating Data withing matrices.

The "Training"
- We use Forward Propagation with a small sample Batch of Data to give the Model a general understanding of the Data. Rawdata input gets converted to output. Data gets transformed based on Weighted summs and Activation functions without changing the wheigths of the model. Each Neuron calculates the weighted sum of the input and uses the activation function to gage if the neuron gets fired in the next layer. Math consists of basic Matrics-Vector-Multiplication
- In the Backward Propagation Process we compute gradients and update weights to minimize loss. It uses gradient descent to adjust weights by moving in the direction of the steepest decrease in the loss function. Imagine it as trying to find a local minimum within a multidimensional space. As to say a ball rolling into a valley, where that valley is the local minimum that is called Cost. The weights are then updated, scaled by the learning rate (alpha) 

# CV-DL MNIST Neural Network

This project implements a simple two-layer neural network for digit classification using the MNIST dataset saved in CSV format. The network is built from scratch using NumPy and follows the standard forward/backward propagation flow.

## Setup

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Project Structure

- `NN.ipynb` - main notebook containing the neural network implementation and training workflow.
- `MNIST_CSV/` - folder with MNIST data exported as CSV files.
- `README.MD` - this file.

## Notebook Breakdown

### 1. Imports

```python
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
```

- `pandas` reads the CSV dataset.
- `numpy` is used for all matrix operations and neural network math.
- `torch` is imported but not used in the current notebook; it may be included for future PyTorch comparisons.
- `matplotlib` is available for plotting if needed.

### 2. Data Loading

```python
data = pd.read_csv("MNIST_CSV/mnist_train.csv")
```

This loads the MNIST training CSV into a pandas DataFrame.

### 3. Data Formatting

```python
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
```

- Converts the dataset to a NumPy array.
- Shuffles the rows to randomize examples.
- Splits the data into a small development set and a larger training set.
- Transposes the arrays so each column is one example.
- `Y_*` contains labels, while `X_*` contains input pixel values.

### 4. Parameter Initialization

```python
def init_params():
    W1 = np.random.randn(10, 784)
    b1 = np.random.randn(10, 1)
    W2 = np.random.randn(10, 10)
    b2 = np.random.randn(10, 1)
    return W1, b1, W2, b2
```

- `W1` and `b1` are weights and biases for the first hidden layer.
- `W2` and `b2` are weights and biases for the output layer.
- The first layer maps from 784 input pixels to 10 hidden units.
- The output layer maps from 10 hidden units to 10 classes.

### 5. Activation Functions

```python
def ReLU(Z):
    return np.maximum(0, Z)
```

- ReLU introduces nonlinearity and helps the network learn complex patterns.

```python
def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)
```

- Softmax converts raw output scores into class probabilities.
- The subtraction of `np.max(Z, axis=0, keepdims=True)` improves numerical stability.

### 6. Forward Propagation

```python
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
```

- Computes the hidden layer pre-activation `Z1` and activation `A1`.
- Computes the output layer pre-activation `Z2` and final probabilities `A2`.
- Returns intermediate values needed for backpropagation.

### 6.1 Math Behind the Layers

1. Input layer to hidden layer

- Input `X` has shape `(784, m)` where `m` is the number of examples.
- Weight matrix `W1` has shape `(10, 784)`.
- Bias vector `b1` has shape `(10, 1)`.

The first linear step is:

```text
Z1 = W1.dot(X) + b1
```

This produces `Z1` with shape `(10, m)`.

Then the activation is:

```text
A1 = ReLU(Z1)
```

ReLU is defined as:

```text
ReLU(z) = max(0, z)
```

2. Hidden layer to output layer

- Hidden activation `A1` has shape `(10, m)`.
- Output weight matrix `W2` has shape `(10, 10)`.
- Output bias `b2` has shape `(10, 1)`.

The second linear step is:

```text
Z2 = W2.dot(A1) + b2
```

`Z2` also has shape `(10, m)`.

3. Softmax output

The softmax function converts each column of `Z2` into a probability distribution:

```python
expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
A2 = expZ / np.sum(expZ, axis=0, keepdims=True)
```

For each example `i`, the softmax output is:

```text
A2[:, i] = exp(Z2[:, i]) / sum(exp(Z2[:, i]))
```

This gives `A2` shape `(10, m)` and each column sums to 1.

4. Loss function

The notebook uses softmax with a cross-entropy loss. For one-hot labels `Y_onehot` with shape `(10, m)`, the loss is:

```text
L = -1/m * sum_{i=1..m} sum_{j=1..10} Y_onehot[j,i] * log(A2[j,i])
```

This measures how well the predicted probability `A2` matches the true label.

### 6.2 Backpropagation Math

1. Output layer error

```text
dZ2 = A2 - Y_onehot
```

This is the gradient of the softmax + cross-entropy loss with respect to `Z2`.

2. Output gradients

```text
dW2 = (1/m) * dZ2.dot(A1.T)
db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
```

- `dW2` has shape `(10, 10)`.
- `db2` has shape `(10, 1)`.

3. Hidden layer error

```text
dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
```

- `W2.T` has shape `(10, 10)`.
- `dZ2` has shape `(10, m)`.
- `deriv_ReLU(Z1)` is a mask with shape `(10, m)`.

4. Hidden gradients

```text
dW1 = (1/m) * dZ1.dot(X.T)
db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
```

- `dW1` has shape `(10, 784)`.
- `db1` has shape `(10, 1)`.

These gradients are then used to update the weights and biases.

### 7. One-Hot Encoding

```python
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
```

- Converts label vectors into one-hot matrices.
- Required so the network can compare predicted probabilities against true labels.

### 8. Backpropagation

```python
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2
```

- Computes gradients for the output and hidden layers.
- `dZ2` is the error at the output layer.
- `dW2` and `db2` are gradients for the output weights and biases.
- `dZ1` is the propagated error through the hidden layer.
- `dW1` and `db1` are gradients for the first-layer weights and biases.

### 9. ReLU Derivative

```python
def deriv_ReLU(Z):
    return Z > 0
```

- Returns a mask used to propagate gradients only where the hidden units were active.

### 10. Parameter Update

```python
def update_params(dW1, db1, dW2, db2, W1, b1, W2, b2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2
```

- Applies gradient descent to update the network parameters.
- `alpha` is the learning rate.

### 11. Predictions and Accuracy

```python
def get_predictions(A2):
    return np.argmax(A2, 0)
```

- Selects the class with highest probability for each example.

```python
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
```

- Computes the classification accuracy.

### 12. Training Loop

```python
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dw2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(dW1, db1, dw2, db2, W1, b1, W2, b2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2
```

- Runs the full training process for a fixed number of iterations.
- Prints progress every 10 iterations.
- Returns the final trained weights and biases.

### 13. Training Call

```python
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)
```

- Trains the network on the training set for 500 iterations.
- Uses a learning rate of `0.1`.
- alpha can be changed based on needs, the lower the alpha the slower the training but the more acurate it gets

## Notes

- Simple model that uses a 10 node single layer system

## References

- Michael Nielsen, [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html)
- 3Blue1Brown neural network playlist
  https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

