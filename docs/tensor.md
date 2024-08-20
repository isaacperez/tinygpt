# Tensor
## Overview
The `Tensor` class in TinyGPT serves as the primary data structure for representing multi-dimensional arrays that can participate in automatic differentiation. `Tensor` encapsulates a `Buffer` for storing data, while also handling the computational graph required for backpropagation.

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
tensor = Tensor.ones((2, 3, 1), dtype=DType.float32)

# Reshape the tensor
reshaped_tensor = tensor.reshape((3, 2))

# Expand the tensor dimensions
expanded_tensor = tensor.expand((2, 3, 4))

# Permute the dimensions
permuted_tensor = tensor.permute((1, 0, 2))
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

The creation of this graph involves a collaborative process between `Tensor` objects, the `apply_op` function, and `GradientFunction` instances. Here’s how they work together to build the graph:

  - __Tensors__: Each tensor serves as a node in the computational graph. When a tensor is created as a result of an operation, it stores a reference to the operation that produced it in its `grad_fn` attribute. This reference is crucial for linking tensors together in the graph, establishing the relationships between input tensors, the operation performed, and the resulting output tensor.

  - __`apply_op` function__: This function orchestrates the execution of operations on tensors and is key to graph construction. When an operation (such as addition or multiplication) is performed on tensors, `apply_op` is called to:
    - Perform the operation by applying the corresponding mathematical or logical function.
    - Generate a new tensor that holds the result of the operation.
    - Attach a `GradientFunction` to the new tensor if it requires gradient tracking, ensuring that the operation and its inputs are recorded in the graph.

  - __`GradientFunction`__: A `GradientFunction` acts as a bridge between the operation that created a tensor and the input tensors involved in that operation. It stores the operation's details and the input tensors, forming a link in the computational graph. 

Together, these components construct the computational graph by linking tensors and operations in a chain of dependencies. As each operation is performed, the graph grows, with each new tensor and its associated `GradientFunction` extending the graph and recording the history of operations. This structure ensures that, when backpropagation is initiated, the graph provides a clear path for calculating gradients and updating model parameters effectively.

#### Example: Creating a Computational Graph
Let's illustrate how a simple operation involving tensors builds the computational graph:

```python
from tinygpt.tensor import Tensor
from tinygpt.utils import DType

# Create two tensors with gradient tracking enabled
a = Tensor([1.0, 2.0, 3.0], dtype=DType.float32, requires_grad=True)
b = Tensor([4.0, 5.0, 6.0], dtype=DType.float32, requires_grad=True)

# Perform operations on the tensors
c = a + b  # Tensor c is created as the result of an addition
d = c * 2.0  # Tensor d is created as the result of a multiplication
```

In this example:
1. __Tensors `a` and `b`__: These are the initial nodes in the computational graph. Since `requires_grad=True`, they will track all operations applied to them.

2. __Operation 1 (Addition)__: When `a + b` is performed, the `apply_op` function is called, which:
    - Creates an instance of the `Add` operation.
    - Executes the addition to produce a new buffer.
    - Creates a new tensor `c` to hold the result.
    - Assigns a `GradientFunction` to `c` that records the `Add` operation and the input tensors (`a` and `b`).

    At this point, the graph has the following nodes:
    - Tensor `a` and Tensor `b`: Input nodes.
    - `GradientFunction` (`Add`): A node representing the addition operation that links `a` and `b` to `c`.
    - Tensor `c`: The output node from the addition.

3. __Operation 2 (Multiplication)__: When `c * 2.0` is performed:
    - The `apply_op` function is invoked again to handle the multiplication.
    - A `Mul` operation instance is created and executed, producing a new buffer.
    - A new tensor `d` is created to store the result.
    - A `GradientFunction` is assigned to `d`, linking it to the multiplication operation and its input tensor `c`.

    Now, the graph is extended to include:
    - Tensor `d`: The output node from the multiplication.
    - `GradientFunction` (`Mul`): A node representing the multiplication operation that connects `c` to `d`.

The computational graph now represents the sequence of operations that produced `d` from `a` and `b`, tracking all necessary dependencies for backpropagation:

```
  Tensor a   Tensor b
     |          |
     |          |
    Add Operation (c = a + b)
           |
           |
    Tensor c
           |
           |
  Mul Operation (d = c * 2.0)
           |
           |
    Tensor d
```

#### The Role of `apply_op` in Graph Construction
Each time an operation is performed on tensors, `apply_op` handles the following steps:
  - __Operation Initialization__: `apply_op` first creates an instance of the operation (e.g., `Add`, `Mul`) being applied.
  - __Forward Pass Execution__: It then performs the forward pass of the operation using the input tensors, producing an output buffer.
  - __Tensor Creation__: A new tensor is created to hold the result of the operation. If any of the input tensors have `requires_grad=True`, the output tensor will also require gradients.
  - __`GradientFunction` Assignment__: If the output tensor requires gradients, `apply_op` assigns a `GradientFunction` to the tensor’s `grad_fn` attribute. This `GradientFunction` tracks the operation and its inputs, effectively linking this tensor to its predecessors in the computational graph.

The implementation of this function can be found towards the end of the `Tensor` class source code [here](../src/tinygpt/tensor.py).

#### Understanding `GradientFunction`
The `GradientFunction` class is a crucial component in TinyGPT’s backpropagation mechanism. It acts as the connective tissue between the operations performed on tensors and the subsequent gradient calculations. Here's an overview of its key responsibilities:

  - __Operation Tracking__: `GradientFunction` records the operation that produced a tensor, along with the input tensors involved. This information is essential for reconstructing the sequence of operations during backpropagation.

  - __Gradient Management__: During backpropagation, `GradientFunction` is responsible for calculating the gradients of the operation's output with respect to its inputs and then propagating these gradients backward through the graph.

When an operation is performed and results in a new tensor, the following steps occur:
1. __Initialization__: The `apply_op` function assigns a `GradientFunction` instance to the `grad_fn` attribute of the new tensor if `requires_grad=True`. This instance stores a reference to the operation and its input tensors.
2. __Backward Pass__: During backpropagation, the `backward()` method of the `GradientFunction` is invoked. This method:
  - __Version checking__: It first verifies that the versions of the input tensors match the versions recorded when the operation was initially performed. This check ensures that no in-place modifications have occurred, which could invalidate the computational graph.
  - __Gradient calculation__: It then computes the gradients of the operation with respect to its inputs using the chain rule. 
  - __Gradient propagation__: Finally, the computed gradients are propagated to the input tensors, enabling the backward pass to continue through the graph. 

3. __Signal Propagation__: Before the actual gradient computation, a "reference signal" is propagated through the graph. This signal ensures that each tensor only propagates its gradient once it has received all expected gradients. This process is managed by the `_propagate_reference_signal()` method and is crucial for optimizing the backpropagation process by reducing redundant computations.

The implementation of this class can be found towards the end of the `Tensor` class source code [here](../src/tinygpt/tensor.py).

### Backpropagation
Backpropagation is a fundamental process in deep learning that involves computing the gradients of the loss function with respect to each tensor in the computational graph. The `Tensor` class in TinyGPT is designed to handle this process efficiently, leveraging a combination of signal propagation, gradient accumulation, and gradient propagation to ensure that gradients are correctly computed and applied.

When backpropagation is initiated, the process involves several key steps:

- **Gradient Computation**: The gradient of the loss function with respect to each input tensor is calculated using the chain rule. This is implemented in the `Tensor` class through the `backward()` method, which computes the local gradients using the `backward()` method of each operation's `GradientFunction`.

- **Gradient Accumulation**: These computed gradients are accumulated in the `grad` attribute of each input tensor, ensuring that all contributions from different operations are combined. The `_accumulate_gradient()` method in the `Tensor` class handles this process, ensuring that gradients are summed up as they flow back through the graph.

- **Gradient Propagation**: The accumulated gradients are then propagated backward through the graph, from the output back to the input tensors. The `_propagate_gradient()` method ensures that gradients are only propagated once all expected gradients have been received, reducing redundant computations.

Each operation's `backward()` method is crucial in this process. It is responsible for computing the local gradients (i.e., the gradients of the operation's output with respect to its inputs) and propagating these gradients backward through the computational graph. 

During backpropagation, the `GradientFunction` plays a vital role by orchestrating how gradients are propagated through the computational graph. Each `Tensor` that requires a gradient is associated with a `GradientFunction`, which keeps track of the operation that created the tensor and its inputs. This function ensures that gradients are correctly accumulated and propagated back through the graph, respecting the dependencies and sequence of operations. 

#### How Backpropagation Works

1. __Starting the Backward Pass__:
    Backpropagation begins at the tensor where the loss is computed. Typically, this tensor is a scalar (i.e., a tensor with zero dimensions). The `backward()` method is called on this tensor to initiate the gradient computation.
    ```python
    loss.backward()
    ```
    If the tensor is a scalar, its gradient is initialized to `1.0`, since the derivative of a value with respect to itself is `1.0`. If the tensor is not a scalar, an external gradient must be provided.

2. __Signal Propagation__:
    Before any actual gradient computation takes place, the system propagates a "signal" through the computational graph. This signal helps each tensor keep track of how many gradients it should expect to receive from its downstream operations. The purpose of this step is to ensure that a tensor only propagates its gradient backward once it has received all the expected gradients. This reduces redundant gradient propagations and makes the backpropagation process more efficient.

    - __Signal Propagation Implementation in `Tensor`__:
        When the `backward()` method is called, the tensor first calls `_propagate_reference_signal()`. This method increments a counter (`_pending_gradients_count`) in each tensor to track the number of gradients it needs to receive. Only after all expected gradients have been received will the tensor propagate its accumulated gradient further back through the graph.

    - __Signal Propagation Implementation in `GradientFunction`__:
        Each `GradientFunction` instance is responsible for propagating this signal to its input tensors. The `_propagate_reference_signal()` method in `GradientFunction` ensures that the signal is only propagated once, preventing redundant increments. This method calls `_increment_pending_gradients()` on each input tensor, signaling that these tensors should expect a gradient from the current operation.

3. __Gradient Initialization__:
    Once the signal propagation is complete, the actual gradient computation begins. The gradient for the starting tensor (typically the loss) is initialized. For scalar tensors, this gradient is `1.0`. For non-scalar tensors, the provided gradient is validated and used for the backward pass.

4. __Gradient Accumulation__:
    As gradients flow backward through the graph, each tensor accumulates the incoming gradients. This is especially important when a tensor contributes to multiple operations, as it needs to sum the gradients from all those operations before sending its accumulated gradient backward.

    - __Gradient Accumulation Implementation in `Tensor`__:
        The `_accumulate_gradient()` method in the `Tensor` class is responsible for accumulating gradients in the `grad` attribute (for leaf tensors or when retaining gradients) and in the `_accumulated_gradient_to_propagate attribute`, which is used to store gradients temporarily until the tensor has received all the expected gradients.

    - __Gradient Accumulation Implementation in `GradientFunction`__:
        The `GradientFunction` itself does not directly accumulate gradients, but it plays a crucial role in ensuring that gradients are propagated correctly. When the `backward()` method of a `GradientFunction` is called, it computes the gradients for its operation and passes these gradients to the input tensors. Each input tensor will then use `_accumulate_gradient()` to store and accumulate the gradients as necessary.

5. __Gradient Propagation__:
    Once a tensor has accumulated all the gradients it expects, it propagates the accumulated gradient to its input tensors. This step involves calling the `backward()` method of the `GradientFunction` associated with the tensor, which further propagates the gradient to the tensor's inputs.

    - __Gradient Propagation Implementation in `Tensor`__:
        The `_propagate_gradient()` method in the `Tensor` class is responsible for this step. It ensures that the gradient is only propagated once all expected gradients have been received, thereby avoiding unnecessary recomputation.

    - __Gradient Propagation Implementation in `GradientFunction`__:
        The `backward()` method in `GradientFunction` is where the actual gradient computation for each operation occurs. It first checks if the versions of the input tensors match the versions recorded when the operation was initially performed, ensuring the computational graph's integrity. Then, it computes the gradients of the operation with respect to its inputs and propagates these gradients to the input tensors. This propagation is recursive, meaning that it continues through the graph until it reaches the leaf tensors.

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

### Visualization of the Computational Graph
You can visualize the computational graph using the `print_dag()` method provided by the `Tensor` class. This method prints the directed acyclic graph of operations leading to the current tensor, helping you to debug and understand how the computations are structured.

```python
x.print_dag()
```
### Zeroing Gradients
Before performing a new forward and backward pass, it’s often necessary to reset the gradients. The `zero_grad()` method resets the gradients to `None`:

```python
a.zero_grad()
```

### Handling Non-Leaf Tensors
By default, gradients are only stored for leaf tensors (tensors that are not the result of an operation). This helps conserve memory during backpropagation. However, if you want to retain gradients for non-leaf tensors (e.g., for further analysis), you can use the `retain_grad()` method.

```python
non_leaf_tensor.retain_grad()
```