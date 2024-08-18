# Welcome to the TinyGPT documentation!
Here you will find documentation for TinyGPT, as well as some examples and tutorials.

## Project overview
The idea behind TinyGPT is to create simple abstractions that allow us to train a GPT model with minimal complexity. 
Only the essential operations necessary for training and using the model have been implemented, making TinyGPT a 
lightweight and easy-to-understand deep learning library.

### Core components
The basic element required in any deep learning library, including TinyGPT, is a data structure to store both the 
model’s weights and the input data. In TinyGPT, this is achieved through the [Buffer](../src/tinygpt/buffer.py) class, 
which acts as a fundamental data structure in the library. The `Buffer` class is designed to represent multidimensional 
arrays, supporting the storage and manipulation of data in various dimensions and essential operations such as addition, 
multiplication, or reshaping, similar to those found in libraries like NumPy. However, `Buffer` is purely a low-level array structure and does not inherently 
support machine learning-specific tasks such as gradient computation or backpropagation.

To handle gradient computation we use the [Operation](../src/tinygpt/mlops.py) class, which details how to 
calculate the derivatives of operations with respect to their inputs, enabling the computation of gradients during 
backpropagation.

To facilitate the use of these operations in a deep learning context, the [Tensor](../src/tinygpt/tensor.py) class is introduced.
`Tensor` serves as an abstraction over the `Buffer` class, enabling the creation and manipulation of multi-dimensional 
arrays with support for automatic differentiation. This means that when operations are performed on `Tensor` objects, 
a computational graph is dynamically built. This graph is composed of `Tensor` instances and associated 
[GradientFunction](../src/tinygpt/tensor.py) objects, which represent the operations applied to the tensors.

When it comes to reusing common patterns of operations, TinyGPT allows you to define them in a 
[Module](../src/tinygpt/module.py). A `Module` encapsulates a set of operations (such as layers in a neural network) and 
can be composed to build more complex models without repeating code. Once a model is defined using Module objects, 
it can be trained on data.

Training a model involves using a [Dataset](../src/tinygpt/dataset.py) class, which handles the reading and preprocessing 
of data from disk, ensuring it is in the correct format for the model. The [DatasetHandler](../src/tinygpt/dataset.py) class 
is used to manage the loading of data in batches, shuffle the data, and perform other tasks necessary for effective training.

For text-based datasets, TinyGPT includes a [BPETokenizer](../src/tinygpt/tokenizer.py) to handle the tokenization process, 
converting raw text into a sequence of tokens that the model can process.

The final piece required to train a model is the optimization algorithm, which is implemented using the 
[Optimizer](../src/tinygpt/optimizer.py) class. The optimizer updates the model’s weights based on the computed gradients, 
driving the learning process.

These core abstractions form the foundation of TinyGPT and provide the essential tools needed to train a GPT model.

A detailed description of the core components in the project can be found in the following links:
  - [Buffer](buffer.md).
  - [Operation](operation.md).
  - [Tensor](tensor.md).
  - [GradientFunction](gradientfunction.md).
  - [Module](module.md).
  - [Optimizer](optimizer.md).
  - [Dataset](dataset.md).
  - [DatasetHandler](datasethandler.md).
  - [BPETokenizer](bpetokenizer.md).
