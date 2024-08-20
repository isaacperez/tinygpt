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
[Optimizer](../src/tinygpt/optimizers.py) class. The optimizer updates the model’s weights based on the computed gradients, 
driving the learning process.

These core abstractions form the foundation of TinyGPT and provide the essential tools needed to train a GPT model.

A detailed description of the core components in the project can be found in the following links:
  - [Buffer](buffer.md).
  - [Operation](operation.md).
  - [Tensor](tensor.md).
  - [Module](module.md).
  - [Optimizer](optimizer.md).
  - [Dataset](dataset.md).
  - [BPETokenizer](bpetokenizer.md).

## Getting started
In this section, we'll walk through an example of how to define a simple neural network and train it on a small dataset.

The first step is to import all the necessary components from TinyGPT to train the model:
```python
from tinygpt.tensor import Tensor 
from tinygpt.dataset import Dataset, DatasetHandler
from tinygpt.nn import FullyConnectedLayer
from tinygpt.optimizers import SGD
from tinygpt.losses import CrossEntropyLoss
from tinygpt.utils import DType
```

The dataset we'll use simulates an AND logic gate. This gate has two binary inputs and one output, which is only active 
when both inputs are 1. We will modify this scenario slightly to use a cross-entropy loss function for training the model. 
Specifically, we'll use two outputs, one for each possible state (0 or 1). By applying softmax to these outputs, we can 
determine whether the gate is in state 0 or 1.

To construct this dataset, we need to implement the `__getitem__` and `__len__` methods of the `Dataset` class in a new class. 
Once the dataset is created, we can use `DatasetHandler` to manage batch creation.

```python
class ANDGateDataset(Dataset):

    def __init__(self) -> None:
        # All posible combinations of inputs (x1, x2) and the corresponding output (y0, y1).
        self.data = [((0, 0), (1, 0)), ((0, 1), (1, 0)), ((1, 0), (1, 0)), ((1, 1), (0, 1))]

    def __getitem__(self, idx: int) -> tuple[tuple[int, int], tuple[int, int]]:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)


dataset = ANDGateDataset()
dataset_handler = DatasetHandler(dataset, batch_size=2, shuffle=True)
```

Next, we define our model. For this problem, a simple single-layer neural network will suffice:

```python
model = FullyConnectedLayer(input_dims=2, output_dims=2, bias=True)
```

To train the model, we need to define the optimizer and the loss function:

```python
optimizer = SGD(model, learning_rate=0.1)
loss_fn = CrossEntropyLoss()
```

The final step is to define the training loop, where the entire dataset is processed by the model to generate predictions. 
These predictions are then used to calculate the loss, and the model's weights are adjusted based on the error:
```python
num_epochs = 1000
for epoch in range(num_epochs):
    for it, batch in enumerate(dataset_handler):

        # Convert the data into Tensors
        xs, ys = batch 
        xs = Tensor(xs, dtype=DType.float32)
        ys = Tensor(ys, dtype=DType.float32)

        # Clean the gradients
        optimizer.zero_grad()

        # Do the forward pass
        pred = model(xs)

        # Calculate the loss
        loss = loss_fn(logits=pred, labels=ys)
        mean_loss = loss.mean(axes=(0,))

        # Do the backward pass
        mean_loss.backward()

        # Apply the gradients
        optimizer.update()
```

Once the model is trained, we can evaluate its performance. To do this, we apply a threshold to the model's output to 
make predictions:
```python
threshold = 0.5
correct_answers = 0
model.eval()
for (x1, x2), (y0, y1) in dataset:
    # Get the output of the model and apply the threshold to get predictions
    model_output = model(Tensor([x1, x2], dtype=DType.float32))
    model_output = model_output.softmax(axis=0).to_python()
    prediction = [int(val > threshold) for val in model_output]

    # Count the correct answers
    if prediction[0] == y0 and prediction[1] == y1:
        correct_answers += 1

print(f"Model accuracy: {correct_answers / len(dataset) * 100:.2f}%")
```

The complete script with all the code is as follows:
```python
from tinygpt.tensor import Tensor 
from tinygpt.dataset import Dataset, DatasetHandler
from tinygpt.nn import FullyConnectedLayer
from tinygpt.optimizers import SGD
from tinygpt.losses import CrossEntropyLoss
from tinygpt.utils import DType


class ANDGateDataset(Dataset):

    def __init__(self) -> None:
        # All posible combinations of inputs (x1, x2) and the corresponding output (y0, y1).
        self.data = [((0, 0), (1, 0)), ((0, 1), (1, 0)), ((1, 0), (1, 0)), ((1, 1), (0, 1))]

    def __getitem__(self, idx: int) -> tuple[tuple[int, int], tuple[int, int]]:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)


dataset = ANDGateDataset()
dataset_handler = DatasetHandler(dataset, batch_size=2, shuffle=True)
model = FullyConnectedLayer(input_dims=2, output_dims=2, bias=True)
optimizer = SGD(model, learning_rate=0.1)
loss_fn = CrossEntropyLoss()

# Training
print("Training...")
num_epochs = 1000
for epoch in range(num_epochs):
    for it, batch in enumerate(dataset_handler):

        # Convert the data into Tensors
        xs, ys = batch 
        xs = Tensor(xs, dtype=DType.float32)
        ys = Tensor(ys, dtype=DType.float32)

        # Clean the gradients
        optimizer.zero_grad()

        # Do the forward pass
        pred = model(xs)

        # Calculate the loss
        loss = loss_fn(logits=pred, labels=ys)
        mean_loss = loss.mean(axes=(0,))

        # Do the backward pass
        mean_loss.backward()

        # Apply the gradients
        optimizer.update()

        print(f"[Epoch {epoch + 1}/{num_epochs}[it. {it + 1}/{len(dataset_handler)}] {mean_loss.to_python():.4f}")

# Inference
print("Inference...")
threshold = 0.5
correct_answers = 0
model.eval()
for (x1, x2), (y0, y1) in dataset:
    # Get the output of the model and apply the threshold to get predictions
    model_output = model(Tensor([x1, x2], dtype=DType.float32))
    model_output = model_output.softmax(axis=0).to_python()
    prediction = [int(val > threshold) for val in model_output]
        
    print(
        f"For input {(x1, x2)} the model predicts {prediction} and the expected output is {(y0, y1)}. "
        f"Model output was [{model_output[0]:.4f}, {model_output[1]:.4f}]"
    )

    # Count the correct answers
    if prediction[0] == y0 and prediction[1] == y1:
        correct_answers += 1

print(f"Model accuracy: {correct_answers / len(dataset) * 100:.2f}%")
```

During training, the average loss settled around `0.025`, and the model achieved 100% accuracy on the AND gate dataset.