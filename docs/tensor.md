# Tensor
## Overview
The `Tensor` class in TinyGPT serves as the primary data structure for representing multi-dimensional arrays that can participate in automatic differentiation. `Tensor` encapsulates a Buffer for storing data, while also handling the computational graph required for backpropagation.

Tensors are designed to be flexible and powerful, supporting various mathematical operations, broadcasting, and gradient tracking. These features enable the `Tensor` class to be seamlessly integrated into the training process of deep learning models.

You can find the implementation of the `Tensor` class in the TinyGPT source code [here](../src/tinygpt/tensor.py).

## Key Features
### Data Representation
A `Tensor` stores its data in a `Buffer`, which is a flattened one-dimensional array. This approach allows for efficient memory usage and manipulation while supporting multi-dimensional operations.

### Gradient Tracking
One of the critical features of the `Tensor` class is its ability to track gradients during the forward and backward passes. This is essential for training neural networks using gradient descent and other optimization algorithms. When `requires_grad` is set to `True`, the tensor will track all operations applied to it and store the gradients needed for backpropagation.
    
### Broadcasting
The `Tensor` class supports broadcasting, allowing operations between tensors of different shapes as long as their dimensions are compatible. 

### Mathematical Operations
`Tensor` objects support a wide range of element-wise mathematical operations, including addition, subtraction, multiplication, division, exponentiation, and more. These operations are designed to work seamlessly with gradient tracking, enabling their use in training models.

### Automatic Differentiation
`Tensor` integrates seamlessly with the `Operation` and `GradientFunction` classes to enable automatic differentiation. When an operation is performed on a tensor, it records the necessary steps in a computational graph. This graph keeps track of how each tensor was derived from others, allowing TinyGPT to efficiently compute gradients during the backward pass.

## Creating Tensors
### Basic Initialization
You can create a `Tensor` directly from Python lists, tuples, or other scalar values. When creating a tensor, you can specify whether it should track gradients by setting the `requires_grad` flag.

```python
from tinygpt.tensor import Tensor
from tinygpt.utils import DType

# Create a Tensor from a list of values
tensor = Tensor([1.0, 2.0, 3.0], dtype=DType.float32, requires_grad=True)
```

### Factory Methods
The `Tensor` class provides several factory methods for creating tensors initialized with specific values:
  - `Tensor.uniform(shape, low=0.0, high=1.0)`: Creates a tensor with values sampled uniformly from the range `[low, high]`.
  - `Tensor.normal(shape)`: Creates a tensor with values sampled from a standard normal distribution.
  - `Tensor.zeros(shape)`: Creates a tensor filled with zeros.
  - `Tensor.ones(shape)`: Creates a tensor filled with ones.

Example:

```python
tensor = Tensor.ones((3, 3), dtype=DType.float32)
```

## Operations
### Mathematical Operations
`Tensors` support a variety of mathematical operations, which can be performed element-wise. These operations automatically track gradients if the tensor's `requires_grad` attribute is `True`.

```python
a = Tensor([1.0, 2.0, 3.0], dtype=DType.float32, requires_grad=True)
b = Tensor([4.0, 5.0, 6.0], dtype=DType.float32)

# Element-wise addition
c = a + b

# Element-wise multiplication
d = a * b

# Element-wise exponentiation
e = a ** 2
```

The Tensor class supports matrix multiplication, also known as the dot product, through the `dot()` method. This operation is crucial for deep learning, especially in layers like fully connected layers or in calculating the output of neural networks.

The `dot()` method performs matrix multiplication between two tensors. It ensures that the inner dimensions of the tensors are compatible for multiplication (i.e., the number of columns in the first tensor must match the number of rows in the second tensor).

Example:

```python
from tinygpt.tensor import Tensor
from tinygpt.utils import DType

# Create two 2D tensors
a = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=DType.float32)
b = Tensor([[5.0, 6.0], [7.0, 8.0]], dtype=DType.float32)

# Perform matrix multiplication (dot product)
c = a.dot(b)

print("Result of dot product:\n", c)
```

### Reduction Operations
Tensors can also perform reduction operations like sum and max, which reduce the dimensions of the tensor by applying the operation along specified axes.

```python
tensor = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=DType.float32)

# Sum across the first axis (rows)
sum_tensor = tensor.sum(axes=(0,))

# Find the maximum value across the second axis (columns)
max_tensor = tensor.max(axes=(1,))
```

### Broadcasting and Shape Manipulation
The `Tensor` class supports broadcasting, allowing you to perform operations on tensors of different shapes, as long as they are compatible.

```python
a = Tensor([1.0, 2.0, 3.0], dtype=DType.float32)
b = Tensor([[1.0], [2.0], [3.0]], dtype=DType.float32)

# Broadcasting addition
c = a + b  # Shape of c will be (3, 3)
```

You can also reshape, expand, and permute tensors to change their shape without altering the underlying data.

```python
tensor = Tensor.ones((2, 3), dtype=DType.float32)

# Reshape the tensor
reshaped_tensor = tensor.reshape((3, 2))

# Expand the tensor dimensions
expanded_tensor = tensor.expand((2, 3, 4))

# Permute the dimensions
permuted_tensor = tensor.permute((1, 0))
```

## Serialization
### Saving and Loading Tensors
You can serialize a tensor to a string and later deserialize it using the following methods:

```python
serialized_tensor = tensor.serialize_tensor()
print(serialized_tensor)  # Output: Serialized string representation

# Deserialize a tensor
deserialized_tensor = Tensor.deserialize_tensor(serialized_tensor)
```

## Computational Graph and Backpropagation
### Building the Computational Graph
In TinyGPT, a computational graph is constructed dynamically as operations are performed on `Tensor` objects. This graph is a directed acyclic graph (DAG) where:
  - __Nodes__ represent tensors and the operations that produce them.
  - __Edges__ represent the flow of data from one operation to the next.

When you perform an operation on tensors, a new tensor is created to hold the result. If any of the input tensors require gradients (i.e., `requires_grad=True`), the resulting tensor will also require gradients, and a new node will be added to the computational graph. This node will be connected to the input tensors, and the operation that produced the result will be stored in the `grad_fn` attribute of the resulting tensor.

For example:

```python

a = Tensor([1.0, 2.0, 3.0], dtype=DType.float32, requires_grad=True)
b = Tensor([4.0, 5.0, 6.0], dtype=DType.float32, requires_grad=True)

c = a + b  # A node is added to the computational graph
d = c * 2  # Another node is added
```

In this example:
  - The addition of `a` and `b` creates a new tensor `c`. The `grad_fn` of `c` will store the operation (`Add`) and the input tensors (`a` and `b`).
  - The multiplication of `c` by `2` creates another tensor `d`, with its `grad_fn` storing the `Mul` operation and the tensor `c`.

To gain a deeper understanding of how tensors and `GradientFunctions` are connected through tensor operations to create the computational graph, be sure to read the section on [apply_op](#apply_op-function-connecting-operations-to-the-computational-graph) and the documentation about `GradientFunction` [here](./gradientfunction.md).


### Backpropagation
Backpropagation is a fundamental process in deep learning that involves computing the gradients of the loss function with respect to each tensor in the computational graph. The `Tensor` class in TinyGPT is designed to handle this process efficiently, leveraging a combination of signal propagation, gradient accumulation, and gradient propagation to ensure that gradients are correctly computed and applied.

When backpropagation is initiated, the process involves several key steps:

- **Gradient Computation**: The gradient of the loss function with respect to each input tensor is calculated using the chain rule. This is implemented in the `Tensor` class through the `backward()` method, which computes the local gradients using the `backward()` method of each operation's `GradientFunction`.

- **Gradient Accumulation**: These computed gradients are accumulated in the `grad` attribute of each input tensor, ensuring that all contributions from different operations are combined. The `_accumulate_gradient()` method in the `Tensor` class handles this process, ensuring that gradients are summed up as they flow back through the graph.

- **Gradient Propagation**: The accumulated gradients are then propagated backward through the graph, from the output back to the input tensors. The `_propagate_gradient()` method ensures that gradients are only propagated once all expected gradients have been received, reducing redundant computations.

Each operation's `backward()` method is crucial in this process. It is responsible for computing the local gradients (i.e., the gradients of the operation's output with respect to its inputs) and propagating these gradients backward through the computational graph. For more details on how operations handle this process, refer to the [Operation class documentation](./operation.md).

During backpropagation, the `GradientFunction` plays a vital role by orchestrating how gradients are propagated through the computational graph. Each `Tensor` that requires a gradient is associated with a `GradientFunction`, which keeps track of the operation that created the tensor and its inputs. This function ensures that gradients are correctly accumulated and propagated back through the graph, respecting the dependencies and sequence of operations. You can read more about the `GradientFunction` class [here](./gradientfunction.md).

#### How Backpropagation Works

1. __Starting the Backward Pass__:
    Backpropagation begins at the tensor where the loss is computed. Typically, this tensor is a scalar (i.e., a tensor with zero dimensions). The `backward()` method is called on this tensor to initiate the gradient computation.
    ```python
    loss.backward()
    ```
    If the tensor is a scalar, its gradient is initialized to `1.0`, since the derivative of a value with respect to itself is `1.0`. If the tensor is not a scalar, an external gradient must be provided.

2. __Signal Propagation__:
    Before any actual gradient computation takes place, the system propagates a "signal" through the computational graph. This signal helps each tensor keep track of how many gradients it should expect to receive from its downstream operations. The purpose of this step is to ensure that a tensor only propagates its gradient backward once it has received all the expected gradients. This reduces redundant gradient propagations and makes the backpropagation process more efficient.

    - __Signal Propagation Implementation__:
        When the `backward()` method is called, the tensor first calls `_propagate_reference_signal()`. This method increments a counter (`_pending_gradients_count`) in each tensor to track the number of gradients it needs to receive. Only after all expected gradients have been received will the tensor propagate its accumulated gradient further back through the graph.

3. __Gradient Initialization__:
    Once the signal propagation is complete, the actual gradient computation begins. The gradient for the starting tensor (typically the loss) is initialized. For scalar tensors, this gradient is `1.0`. For non-scalar tensors, the provided gradient is validated and used for the backward pass.

4. __Gradient Accumulation__:
    As gradients flow backward through the graph, each tensor accumulates the incoming gradients. This is especially important when a tensor contributes to multiple operations, as it needs to sum the gradients from all those operations before sending its accumulated gradient backward.

    - __Gradient Accumulation Implementation__:
        In the `Tensor` class, the `_accumulate_gradient()` method is responsible for accumulating gradients in the `grad` attribute (for leaf tensors or when retaining gradients) and in the `_accumulated_gradient_to_propagate` attribute, which is used to store gradients temporarily until the tensor has received all the expected gradients.

5. __Gradient Propagation__:
    Once a tensor has accumulated all the gradients it expects, it propagates the accumulated gradient to its input tensors. This step involves calling the `backward()` method of the `GradientFunction` associated with the tensor, which further propagates the gradient to the tensor's inputs.

    - __Gradient Propagation Implementation__:
        The `_propagate_gradient()` method in the `Tensor` class is responsible for this step. It ensures that the gradient is only propagated once all expected gradients have been received, thereby avoiding unnecessary recomputation.

6. __Releasing the Computational Graph__:
    After backpropagation, the computational graph is usually released to free up memory. This means that you cannot perform another backward pass unless you explicitly retain the graph by passing `retain_graph=True` to the `backward()` method.

    ```python
    loss.backward(retain_graph=True)
    ```

#### Example: Backpropagation in Action
Here’s a simple example demonstrating how the computational graph is built and how backpropagation is performed:

```python

from tinygpt.tensor import Tensor
from tinygpt.utils import DType

# Create input tensors
x = Tensor([1.0, 2.0, 3.0], dtype=DType.float32, requires_grad=True)
y = Tensor([4.0, 5.0, 6.0], dtype=DType.float32, requires_grad=True)

# Perform operations
z = x * y  # Element-wise multiplication
loss = z.sum(axes=(0,))  # Sum all elements to get a scalar loss

# Perform backpropagation
loss.backward()

# Print gradients
print("Gradient of x:", x.grad)
print("Gradient of y:", y.grad)
```

#### Visualization of the Computational Graph
You can visualize the computational graph using the `print_dag()` method provided by the `Tensor` class. This method prints the directed acyclic graph of operations leading to the current tensor, helping you to debug and understand how the computations are structured.

```python
x.print_dag()
```
#### Zeroing Gradients
Before performing a new forward and backward pass, it’s often necessary to reset the gradients. The `zero_grad()` method resets the gradients to `None`:

```python
a.zero_grad()
```

#### Handling Non-Leaf Tensors
By default, gradients are only stored for leaf tensors (tensors that are not the result of an operation). This helps conserve memory during backpropagation. However, if you want to retain gradients for non-leaf tensors (e.g., for further analysis), you can use the `retain_grad()` method.

```python
non_leaf_tensor.retain_grad()
```

### `apply_op` Function: Connecting Operations to the Computational Graph
The `apply_op` function is a key component in TinyGPT that facilitates the connection between tensor operations and the construction of the computational graph. This function is responsible for:

  - __Creating the Operation object__: It instantiates the specific operation class (e.g., `Add`, `Mul`, `Dot`) that defines the forward and backward passes for the operation.

  - __Performing the operation__: It applies the operation to the input tensors by calling the operation's `forward()` method.

  - __Creating the resulting tensor__: It wraps the result of the operation in a new `Tensor` object.

  - __Setting up the computational graph__: If any of the input tensors require gradients, `apply_op` sets up the necessary connections in the computational graph by assigning a `GradientFunction` to the resulting tensor.

The implementation of this function can be found towards the end of the `Tensor` class source code [here](../src/tinygpt/tensor.py).

#### How apply_op Works

When you perform an operation on tensors (e.g., `+`, `*`, `dot()`), the corresponding magic method or operation method in the `Tensor` class calls `apply_op`. Here’s a breakdown of what happens inside `apply_op`:

  -__Determining Gradient Requirement__:
    The function first checks whether any of the input tensors require gradients. If so, the resulting tensor will also require gradients, and `requires_grad` is set to `True`.

  - __Instantiating the Operation Object__:
    The function then creates an instance of the operation class (passed as `operation_cls`). This object will handle the forward and backward computations.

  - __Performing the Forward Pass__:
    `apply_op` calls the `forward()` method of the operation object, passing the buffers of the input tensors. This method returns a new buffer containing the result of the operation.

  - __Creating the Output Tensor__:
    A new Tensor is created to hold the result of the operation, using the buffer returned by the forward pass.

  - __Setting Up the Computational Graph__:
    If the output tensor requires gradients, `apply_op` creates a `GradientFunction` object, linking the operation and the input tensors. This `GradientFunction` is assigned to the `grad_fn` attribute of the resulting tensor, making it a part of the computational graph.

  - __Returning the Resulting Tensor__:
    Finally, the resulting tensor is returned, now connected to the computational graph if gradients are needed.

#### Example of apply_op in Action
Here’s an example of how `apply_op` is used within a tensor operation:

```python
def __add__(self, other: Any) -> Tensor:
    if not isinstance(other, Tensor):
        other = Tensor(other)
    return apply_op(mlops.Add, *self._broadcasted(other))
```

In this example, when two tensors are added using the `+` operator:

  - The `__add__` method is called.
  - `apply_op` is invoked with the `Add` operation class and the two tensors as arguments.
  - `apply_op` handles the creation of the computational graph and returns the resulting tensor.

#### How apply_op Helps Create the Computational Graph
Every time you perform an operation on tensors that requires gradient tracking, `apply_op` ensures that the resulting tensor is connected to the computational graph. Here’s how:

  - __Operation Tracking__: Each operation (like addition, multiplication, etc.) is tracked by the creation of an `Operation` object that knows how to compute both the forward result and the backward gradients.

  - __Graph Construction__: The `GradientFunction` created by `apply_op` links the resulting tensor to its inputs and the operation that produced it. This forms a node in the computational graph.

  - __Backward Pass__: During backpropagation, this graph is traversed in reverse, using the connections set up by `apply_op` to compute gradients and propagate them backward through the network.

In essence, `apply_op` is the glue that binds tensor operations to the computational graph, enabling the powerful automatic differentiation capabilities of TinyGPT.