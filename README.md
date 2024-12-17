# Neural Network from Scratch

This project demonstrates how to build a neural network from scratch using only NumPy and basic linear algebra. The goal is to classify handwritten digits from the well-known MNIST dataset, replicating a neural network's functionality without relying on deep learning libraries like TensorFlow or PyTorch.

## Inspiration

This implementation is inspired by [Samson Zhang's YouTube video](https://youtu.be/w8yWXqWQYmU?si=84GMJdX_9kyv3SuW), where he explores the principles of constructing a neural network from the ground up.

## Features

- **Custom Neural Network Implementation:** 
  - Two-layer fully connected network.
  - ReLU activation function for hidden layers.
  - Softmax function for multi-class classification.
  - Categorical cross-entropy for loss calculation.
  
- **Data Handling:**
  - Custom MNIST dataset loader implemented using Python's `struct` and `numpy`.
  - Normalization and reshaping of input images to be compatible with the neural network.

- **Learning from Scratch:**
  - Manual implementation of forward propagation, backward propagation, and gradient descent.
  - Parameter initialization, activation functions, and derivatives implemented from first principles.

## How It Works

### Data Loading
The MNIST dataset is loaded using a custom data loader (`MnistDataloader`), which parses the raw `.idx` files into NumPy arrays. The training and test datasets are normalized to a range of [0, 1] for effective learning.

### Neural Network Architecture
- **Layer 1:** 784 input features → 10 neurons.
- **Layer 2:** 10 neurons → 10 neurons (one for each class).
- **Activation Functions:**
  - Hidden Layer: ReLU
  - Output Layer: Softmax
- **Loss Function:** Categorical cross-entropy.

### Training
- **Forward Propagation:** Computes activations and predictions.
- **Backward Propagation:** Calculates gradients using the chain rule.
- **Gradient Descent:** Updates weights and biases based on gradients with a learning rate.

### Key Functions
- `forward_propagation`: Computes activations for each layer.
- `backward_propagation`: Computes gradients for all weights and biases.
- `update_parameters`: Updates model parameters based on gradients.
- `gradient_descent`: Iteratively optimizes the model using the above functions.

### Results
The model is trained for 1000 iterations, achieving a reasonable accuracy given the simplicity of the implementation. The network's performance is evaluated using the accuracy of predictions on the MNIST test set.

## Getting Started

1. Clone this repository.
2. Ensure the MNIST dataset is available in the specified directory structure.
3. Install the required Python libraries (`numpy` and `matplotlib`).
4. Run the provided code to load the dataset, train the model, and evaluate its performance.

## Acknowledgements

Special thanks to Samson Zhang for the inspiration and guidance provided in his video. This project was an excellent exercise in understanding the fundamentals of neural networks.
