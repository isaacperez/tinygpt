# TinyGPT
TinyGPT is a minimalistic library for implementing, training, and performing inference on GPT models from scratch in Python, with no external dependencies. Inspired by [NanoGPT](https://github.com/karpathy/nanoGPT), [Tinygrad](https://github.com/tinygrad/tinygrad), [Pytorch](https://github.com/pytorch/pytorch), and [MLX](https://github.com/ml-explore/mlx), TinyGPT aims to be as educational as possible, avoiding complex optimizations that might obscure the underlying concepts.

## Features
- **Pure Python Implementation**: TinyGPT is written entirely in Python with no external dependencies, making it easy to follow and modify.
- **Didactic Focus**: Prioritizes readability and understanding over optimization, making it an excellent learning tool.
- **Modular Design**: The library is divided into several modules, each focusing on a specific aspect of training and inference.

## Installation
The current recommended way to install TinyGPT is from source.

### From source
```bash
$ git clone https://github.com/isaacperez/tinyGPT.git
$ cd tinygpt
$ python -m pip install -e .
```
Don't forget the `.` at the end!

## Project Structure
  - `buffer.py`: Provides a low-level implementation of array operations, similar to NumPy arrays.
  - `dataset.py`: Handles data loading and preprocessing.
  - `losses.py`: Contains implementations of loss functions used for training.
  - `mlops.py`: Provides a low-level implementation of array operations, similar to NumPy arrays.
  - `module.py`: Defines the base module from which all model components inherit.
  - `nn.py`: Contains the neural network components, including layers and activation functions.
  - `optimizers.py`: Implements optimization algorithms used during training.
  - `tensor.py`: Provides a minimal tensor implementation to support basic tensor operations.
  - `tokenizer.py`: Handles tokenization of text data for model input.
  - `utils.py`: Contains miscellaneous utility functions used across the library.

## Examples
The `examples/` directory contains scripts demonstrating how to use TinyGPT for various tasks:

  - `gpt.py`: A basic example of training and using a GPT model.
  - `mnist.py`: An example of using TinyGPT's neural network components on the MNIST dataset.

To run these examples, navigate to the examples/ directory and execute the Python scripts:

```bash
$ cd examples
$ python gpt.py
$ python mnist.py
```

These examples will guide you through setting up the model, training it, and performing inference, providing a hands-on understanding of how TinyGPT works.

## Documentation
Documentation along with a quick start guide can be found in the `docs/` directory.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

### Testing
You need to install pytest:
```bash
$ python -m pip install pytest
```
and [TinyGPT](#installation), then run: 
```bash
$ pytest
```