# Optimizer
## Overview
The `Optimizer` class in TinyGPT serves as a base class for all optimization algorithms used to adjust the parameters of a neural network during training. Optimizers are responsible for updating the model's parameters based on the gradients computed during backpropagation, helping to minimize the loss function.

This class is designed to be extended by specific optimizer implementations, such as SGD (Stochastic Gradient Descent) and Adam. It provides the core functionalities needed for these implementations, including parameter updates, gradient zeroing, and state management.

You can find the implementation of this class and the optimizers [here](../src/tinygpt/optimizers.py).

## Key Components
1. __`OptimizerState`__

    - A specialized dictionary that recursively creates new instances of itself for missing keys. This is particularly useful for maintaining optimizer states such as momentum or running averages, which are stored per parameter.
    
    - The `OptimizerState` class also includes a get method that, unlike a regular dictionary, sets the key to a default value if it was not already present.

2. __Optimizer Attributes__

    - `state`: An instance of `OptimizerState`, this dictionary holds the internal state of the optimizer, such as momentum vectors or running averages.
    - `module`: The model (Module instance) whose parameters the optimizer will update.

3. __Core Methods__

    - `__init__(module)`: Initializes the optimizer with a given Module instance, setting up the state dictionary.
    - `update()`: Applies the gradients to the model's parameters and updates them using the optimizer's specific algorithm.
    - `apply_single(parameter, state)`: A method to be implemented by subclasses. It defines how a single parameter is updated, given its current state.
    - `zero_grad()`: Resets the gradients of all parameters in the associated model to zero.
    - `load_state(file_or_weights, strict=True)`: Loads the optimizer's state from a JSON file or a list of weight tuples. In strict mode, it ensures the loaded state matches the current model.
    - `save_state(file)`: Saves the current state of the optimizer to a JSON file.

## Specific Optimizers
### SGD (Stochastic Gradient Descent)
The SGD class extends the `Optimizer` class to implement the stochastic gradient descent algorithm with optional momentum, weight decay (L2 regularization), and Nesterov momentum.

#### Key Features

  - __Learning Rate__: The primary hyperparameter controlling the step size in the direction of the gradient.
  - __Momentum__: Helps accelerate gradients vectors in the right direction, leading to faster converging.
  - __Weight Decay__: Adds an L2 penalty to the loss function, which can help prevent overfitting.
  - __Dampening__: Reduces the momentum's contribution to the update.
  - __Nesterov Momentum__: A variant of momentum that considers the gradient of the parameter after it has been updated by the momentum.

#### Example Usage

```python
# Initialize the model and optimizer
model = SimpleModel()
optimizer = SGD(module=model, learning_rate=0.01, momentum=0.9)

# Perform a forward pass, compute the loss, backpropagate, and update the parameters
loss = model(inputs).sum()
loss.backward()
optimizer.update()
```

### Adam Optimizer
The Adam class implements the Adam optimization algorithm, which combines the advantages of both AdaGrad and RMSProp. It computes individual adaptive learning rates for different parameters based on estimates of first and second moments of the gradients.

#### Key Features
  - __Learning Rate__: Controls the step size in the parameter space.
  - __Betas__: Coefficients used for computing running averages of gradient and its square.
  - __Epsilon__: A small constant added to the denominator to improve numerical stability.

#### Example Usage

```python

# Initialize the model and optimizer
model = SimpleModel()
optimizer = Adam(module=model, learning_rate=0.001)

# Perform a forward pass, compute the loss, backpropagate, and update the parameters
loss = model(inputs).sum()
loss.backward()
optimizer.update()
```

## Notes
  - The `Optimizer` class is intended to be subclassed to create different optimization algorithms. SGD and Adam are provided as examples of how to extend the base `Optimizer` class.