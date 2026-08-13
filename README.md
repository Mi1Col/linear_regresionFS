# Single Neuron Linear Regression Form Scratch

A simple single-layer perceptron that learns to perform linear regression using Gradient Descent.

## What is this?

This project explores how a single neuron can learn a linear relationship between input values `x` and target values `y`.

The neuron uses:

$$
\hat{y} = wx + b
$$

where:
- `w` is the weight (slope)
- `b` is the bias (y-intercept)
- `ŷ` is the prediction

The model learns the values of `w` and `b` by minimizing the Mean Squared Error (MSE).

## How does it learn?

The training process is:

1. Make predictions using `ŷ = wx + b`.
2. Calculate the MSE.
3. Calculate the gradients with respect to `w` and `b`.
4. Update the parameters using Gradient Descent.
5. Repeat until the error is minimized.

The parameters are updated using:

$$
w \leftarrow w - \eta \frac{\partial MSE}{\partial w}
$$

$$
b \leftarrow b - \eta \frac{\partial MSE}{\partial b}
$$

where `η` is the learning rate.

## What I wanted to learn

The main goal of this project was to understand the mathematics behind machine learning rather than treating a neural network as a black box.

In particular, I wanted to understand:

- Linear regression
- Neurons and weights
- Loss functions
- Derivatives and partial derivatives
- The chain rule
- Gradient Descent

## Documentation

For a more detailed explanation of the mathematics and the reasoning behind the implementation, see:

**[Understanding Linear Regression Through a Single Neuron](./docs/Understanding_Linear_Regression_Through_a_Single_Neuron.pdf)**

## Project status

This is a learning project created while studying mathematics, programming and machine learning.

More complex neural networks will hopefully come later. :)

## Author

Michenchin
