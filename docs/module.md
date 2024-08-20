# Module
## Overview
The `Module` class in TinyGPT serves as a fundamental building block for constructing and managing deep learning models. It is designed to encapsulate the parameters (tensors) and submodules that form a neural network, providing a convenient interface for tasks such as saving/loading weights, traversing the model's hierarchy, and applying transformations to parameters.

The `Module` class is a subclass of Python's dict, allowing it to store its parameters and submodules in a flexible and hierarchical manner. It also provides several utilities to manage these elements, such as freezing/unfreezing parameters, setting the model to training or evaluation mode, and more.

You can find the implementation of this class and the optimizers [here](../src/tinygpt/module.py).

## Key Features
1. __Hierarchical Structure Management__

    - __Submodules__: The `Module` class can contain other `Module` instances as its children, enabling the construction of complex neural network architectures by composing simpler modules.

    - __Parameter Management__: Parameters (instances of the Tensor class) can be easily managed within a `Module`, including functionalities to retrieve, update, freeze, or unfreeze them.

2. __Training and Evaluation Modes__

    - __Mode Switching__: The `Module` class can be toggled between training and evaluation modes using the `train()` and `eval()` methods. This feature is particularly useful for layers like dropout that behave differently during training and evaluation.

3. __Weight Management__

    - __Saving and Loading Weights__: The `Module` class provides methods to save the model's weights to a JSON file and load them back, ensuring easy model persistence and restoration.

    - __Strict Loading__: The `load_weights` method supports a strict mode that ensures the loaded weights match the model's architecture exactly, with options for more flexible loading when necessary.

4. __Parameter and Submodule Filtering__

    - __Filtering and Mapping__: The `filter_and_map` method allows for complex traversals of the model's parameters and submodules, applying filters and transformations as needed. This is useful for operations like retrieving all trainable parameters or applying a specific function to each submodule.

5. __Freezing and Unfreezing Parameters__

    - __Freeze/Unfreeze__: The freeze and unfreeze methods allow for selective freezing or unfreezing of parameters, which is crucial for transfer learning or fine-tuning parts of a model.

6. __Zeroing Gradients__

    - __Zero Gradients__: The `zero_grad` method zeros out the gradients of all trainable parameters in the module, a common requirement before performing a new optimization step.

## Example Usage
Creating a Simple Model

```python
from tinygpt.module import Module
from tinygpt.tensor import Tensor


class SimpleModel(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Tensor([1.0, 2.0], requires_grad=True)
        self.linear2 = Tensor([3.0, 4.0], requires_grad=True)

    def __call__(self, x):
        return self.linear1 * x + self.linear2

# Initialize the model
model = SimpleModel()

# Set to training mode
model.train()

# Save the model's weights
model.save_weights("model_weights.json")

# Load the model's weights
model.load_weights("model_weights.json")
```

### Freezing and Unfreezing Parameters

```python
# Freeze all parameters
model.freeze()

# Unfreeze only the biases
model.unfreeze(keys="bias")
```

### Applying a Function to All Parameters

```python
# Apply ReLU to all parameters
model.apply(lambda x: x.relu())
```

### Switching Between Training and Evaluation Modes

```python
# Set to evaluation mode
model.eval()

# Set back to training mode
model.train()
```

## Notes
  - The `Module` class is designed to be extended. Most of its functionality comes to life when it's subclassed to create specific models.
  - Custom models can override methods like `_extra_repr()` to provide additional information in the module's string representation.
  - The main submodules used in TinyGPT, including components such as `FullyConnectedLayer`, `MLP`, `Embedding`, `LayerNorm`, `CasualSelfAttention`, `TransformerBlock`, and `GPT`, are implemented in the [nn.py](../src/tinygpt/nn.py) file. This file contains essential building blocks for constructing complex neural network architectures.