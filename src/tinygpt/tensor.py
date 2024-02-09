from __future__ import annotations
import re
from typing import Any, Union, Optional

from tinygpt.utils import DType, print_dag, parse_value
from tinygpt.buffer import Buffer
import tinygpt.mlops as mlops


class Tensor():

    def __init__(self, data: Any, dtype: DType = None, requires_grad: bool = False) -> None:
        # Save the data in a buffer
        if isinstance(data, Buffer):
            self.buffer = data
        else:
            self.buffer = Buffer(data, dtype)

        # Gradient-related metadata
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self.is_leaf = True
        self._version = 0
        self._retain_grad = False

        self._pending_gradients_count = 0
        self._accumulated_gradient_to_propagate = None

        if self.requires_grad and self.dtype != DType.float32:
            raise RuntimeError("Only float32 Tensors can require gradients")

    def __repr__(self) -> str:
        return (
            f"<Tensor {hex(id(self))}: {self.buffer}, shape={self.shape}, dtype={self.dtype}, "
            f"requires_grad={self.requires_grad}>"
        )

    def __str__(self) -> str:
        return f"<Tensor {self.buffer}, shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad}>"

    def print_dag(self) -> None:
        print_dag(self)

    @property
    def shape(self) -> tuple:
        return self.buffer.shape

    @property
    def ndim(self) -> int:
        return self.buffer.ndim

    @property
    def dtype(self) -> DType:
        return self.buffer.dtype

    def _broadcasted(self, other: Tensor) -> tuple[Tensor, Tensor]:
        # Make two tensors broadcastable by reshaping and expanding them as needed
        x, y = self, other

        # Return immediately if shapes are already the same
        if x.shape == y.shape:
            return x, y

        # Adjust the number of dimensions to match
        shape_delta = x.ndim - y.ndim
        if shape_delta > 0:
            # Add extra dimensions to y if x has more dimensions
            y = y.reshape((1,) * shape_delta + y.shape)
        elif shape_delta < 0:
            # Add extra dimensions to x if y has more dimensions
            x = x.reshape((1,) * -shape_delta + x.shape)

        # Check if shapes match after adding dimensions
        if x.shape == y.shape:
            return x, y

        # If shapes still don't match, expand dimensions where sizes differ
        shape_ret = tuple([max(x, y) for x, y in zip(x.shape, y.shape)])
        if x.shape != shape_ret:
            x = x.expand(shape_ret)
        if y.shape != shape_ret:
            y = y.expand(shape_ret)

        return x, y

    def __add__(self, other: Any) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return apply_op(mlops.Add, *self._broadcasted(other))

    def __sub__(self, other: Any) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return apply_op(mlops.Sub, *self._broadcasted(other))

    def __neg__(self):
        return apply_op(mlops.Neg, self)

    def __mul__(self, other: Any) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return apply_op(mlops.Mul, *self._broadcasted(other))

    def __truediv__(self, other: Any) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return apply_op(mlops.Div, *self._broadcasted(other))

    def __pow__(self, exponent: Union[int, float]) -> Tensor:
        if not isinstance(exponent, (int, float)):
            raise TypeError("Only supporting int/float powers for now")

        return apply_op(mlops.Pow, self, exponent=exponent)

    def exp(self) -> Tensor:
        return apply_op(mlops.Exp, self)

    def log(self) -> Tensor:
        return apply_op(mlops.Log, self)

    def maximum(self, other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return apply_op(mlops.Maximum, *self._broadcasted(other))

    def relu(self) -> Tensor:
        return apply_op(mlops.Relu, self)

    def sum(self, axes: tuple, keepdim: bool = False) -> Tensor:
        if keepdim:
            return apply_op(mlops.Sum, self, axes=axes)
        else:
            result = apply_op(mlops.Sum, self, axes=axes)
            return result.reshape(tuple(val for val in result.shape if val != 1))

    def max(self, axes: tuple, keepdim: bool = False) -> Tensor:
        if keepdim:
            return apply_op(mlops.Max, self, axes=axes)
        else:
            result = apply_op(mlops.Max, self, axes=axes)
            return result.reshape(tuple(val for val in result.shape if val != 1))

    def softmax(self, axis: int) -> Tensor:
        if not isinstance(axis, int):
            raise TypeError(f"Expecting int value for axis but found type {type(axis)}")

        if self.ndim == 0:
            raise ValueError("Softmax not implemented for 0-d tensors")

        self_normalized = self - self.max(axes=(axis,), keepdim=True)
        self_exponential = self_normalized.exp()
        summation = self_exponential.sum(axes=(axis,), keepdim=True)

        return self_exponential / summation

    def reshape(self, shape: tuple) -> Tensor:
        return apply_op(mlops.Reshape, self, new_shape=shape)

    def expand(self, shape: tuple) -> Tensor:
        return apply_op(mlops.Expand, self, new_shape=shape)

    def permute(self, dims: tuple) -> Tensor:
        return apply_op(mlops.Permute, self, dims=dims)

    def __eq__(self, other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            raise TypeError(f"Expecting Tensor, but found type {type(other)}")
        return Tensor(self.buffer == other.buffer, requires_grad=False)

    def transpose(self, axis1: int, axis2: int) -> Tensor:
        # Transpose method to swap two dimensions (axes) of a Tensor (both axes can be the same, but it has no effect)

        # Validate that axis1 and axis2 are integers
        if not isinstance(axis1, int) or not isinstance(axis2, int):
            raise TypeError("axis1 and axis2 must be an integer")

        # Check that axis1 and axis2 are within the valid range of the tensor's dimensions
        if any(axis < 0 or axis >= self.ndim for axis in [axis1, axis2]):
            raise ValueError("axis1 and axis2 should be positive and within the dimension range")

        # Generate a sequence representing the order of dimensions
        order = list(range(self.ndim))

        # Swap the positions of axis1 and axis2 in this order
        order[axis1], order[axis2] = order[axis2], order[axis1]

        # Apply the permute operation to reorder the dimensions of the tensor according to the new order
        return self.permute(tuple(order))

    def dot(self, other: Tensor):
        # Perform matrix multiplication (dot product) between this tensor (`self`) and another tensor (`other`)

        # Check if the other operand is a tensor
        if not isinstance(other, Tensor):
            raise TypeError(f"Expecting Tensor, but found type {type(other)}")

        # Ensure both tensors are at least 1-dimensional, as matrix multiplication is not defined for 0D tensors
        if self.ndim == 0 or other.ndim == 0:
            raise RuntimeError(
                f"Both arguments to matmul need to be at least 1D, but they are {self.ndim}D and {other.ndim}D"
            )

        # Check if the inner dimensions are compatible. In matrix multiplication, the number of columns in the first
        # matrix (last dimension of `self`) must be equal to the number of rows in the second matrix
        # (appropriate dimension of `other`)
        if self.shape[-1] != other.shape[-min(other.ndim, 2)]:
            raise RuntimeError(
                f"Input Tensor shapes {self.shape} and {other.shape} cannot be multiplied"
                f"({self.shape[-1]} != {other.shape[-min(other.ndim, 2)]})")

        # Reshape the tensors to make their dimensions compatible for element-wise multiplication
        # This involves adding a singleton dimension before the last dimension of `self` and before the last two
        # dimensions of `other`. It aligns the 'row' dimension of `self` with a new singleton dimension in `other`,
        # and vice versa for the 'col' dimension. This step is necessary to enable broadcasting during the element-wise
        # multiplication.
        x = self.reshape((*self.shape[0:-1], *[1] * min(self.ndim - 1, other.ndim - 1, 1), self.shape[-1]))
        w = other.reshape(
            (*other.shape[0:-2], *[1] * min(self.ndim - 1, other.ndim - 1, 1), *other.shape[-min(other.ndim, 2):])
        )

        # Transpose the last two dimensions of `w` to align it properly with `x` for element-wise multiplication
        dims_of_w = list(range(w.ndim))
        w = w.transpose(dims_of_w[-1], dims_of_w[-min(w.ndim, 2)])

        # Perform element-wise multiplication of the two tensors. This mimics matrix multiplication by multiplying
        # corresponding elements in the aligned dimensions
        mult = x * w

        # Sum over the last dimension to collapse the result into the final matrix multiplication output. This step
        # effectively sums the products of corresponding elements, completing the matrix multiplication process
        return (mult).sum((mult.ndim - 1,))

    @staticmethod
    def uniform(shape: tuple, **kwargs):
        return Tensor(Buffer.uniform(shape), **kwargs)

    @staticmethod
    def zeros(shape: tuple, **kwargs):
        return Tensor(Buffer.zeros(shape), **kwargs)

    @staticmethod
    def ones(shape: tuple, **kwargs):
        return Tensor(Buffer.ones(shape), **kwargs)

    def backward(self, incoming_gradient: Optional[Buffer] = None, retain_graph: bool = False) -> None:
        """
        Perform the backward pass to compute gradients.

        The method first checks if the computational graph has been previously released. If so, and if the tensor is not 
        a leaf, it raises a RuntimeError.

        Steps in the backward pass:
        1. Gradient Initialization: Before any propagation, the incoming gradient is initialized or validated. It 
           ensures that the incoming gradient is correctly formatted and compatible with the tensor's shape and type.

        2. Propagation of reference signal: This initial step sends a notification signal to each tensor and operation 
           involved in the computational graph. This signal helps tensors count the number of operations they are 
           involved in. This information is crucial because a tensor should not send its gradient back through the graph 
           until it has received all expected gradients from subsequent operations. Accumulating gradients is crucial 
           to avoid repeatedly traversing the graph back to the leaf nodes every time a new gradient is received.

        3. Gradient accumulation: The incoming gradient is combined with any existing gradients the tensor may have.
           This is especially important in scenarios where a tensor contributes to multiple operations, as it needs to 
           accumulate gradients from all of them before sending them backward.

        4. Gradient propagation: Once a tensor has received all expected gradients (as determined in step 1), it 
           sends its accumulated gradient backward through the graph. This step involves calling the backward method of 
           the `GradientFunction` associated with the tensor, which further propagates the gradient to the tensor's 
           inputs.
        """
        # Check if the tensor requires gradient computation
        if self.requires_grad:
            # Validate the state of the computational graph
            self._validate_graph_state()

            # Step 1: Initialize the incoming gradient
            incoming_gradient = self._initialize_incoming_gradient(incoming_gradient)

            # Step 2: Propagate reference signal through the graph
            self._propagate_reference_signal()

            # Step 3: Accumulate the incoming gradient
            self._accumulate_gradient(incoming_gradient)

            # Step 4: Propagate the gradient backward through the graph
            self._propagate_gradient(retain_graph)

    def _validate_graph_state(self) -> None:
        """
        Validate the state of the computational graph before performing the backward pass.

        This method checks if the tensor's computational graph is available for backpropagation. It raises an error 
        if attempting to backpropagate through a non-leaf tensor whose graph has already been released (i.e., `grad_fn` 
        is None).
        """
        if not self.is_leaf and self.grad_fn is None:
            raise RuntimeError(
                "Trying to backward through the graph a second time. The graph is freed when you call .backward(). "
                "Specify retain_graph=True if you need to backward through the graph a second time."
            )

    def _initialize_incoming_gradient(self, incoming_gradient: Buffer) -> Buffer:
        """
        Initialize the incoming gradient for the backward pass.

        This method prepares the gradient that will be used for the backward pass, either by initializing it or by 
        validating the provided gradient.

        The initialization process varies based on whether the incoming gradient is provided:
        - If no incoming gradient is provided (`None`), it means this is the starting point of backpropagation. For 
        scalar tensors, the gradient is initialized to 1.0, representing the derivative of the tensor with respect to 
        itself. However, for non-scalar tensors, a gradient must be provided; hence an error is raised.
        - If an incoming gradient is provided, this method checks its validity. The gradient must be a `Buffer` object 
        of dtype `float32` and must have the same shape as the tensor. Any deviation from this results in an error.
        """
        if incoming_gradient is None:
            # Handle the initialization for the first backward call
            if self.ndim == 0:
                # Initialize the gradient as 1.0 for scalar tensors
                return Buffer(1.0)
            else:
                # Backward on non-scalar tensors must have an incoming gradient
                raise RuntimeError("backward can only be called on scalar tensors or with an existing gradient")
        else:
            # Handle subsequent backward calls with provided gradient
            if not isinstance(incoming_gradient, Buffer):
                raise TypeError(f"incoming_gradient is not a Buffer. Found {type(incoming_gradient)}")
            if incoming_gradient.dtype != DType.float32:
                raise TypeError(f"Expecting float32 for dtype of incoming_gradient. Found {incoming_gradient.dtype}")
            if incoming_gradient.shape != self.shape:
                raise RuntimeError(
                    f"incoming_gradient ({incoming_gradient.shape}) doesn't match tensor shape ({self.shape})"
                )

            return incoming_gradient

    def _accumulate_gradient(self, incoming_gradient: Buffer) -> None:
        """
        Accumulate the incoming gradient in preparation for backpropagation.

        This function manages the accumulation of gradients in two scenarios:
        - For leaf tensors (tensors that are not results of an operation), it always accumulates gradients in the `grad` 
          attribute.
        - For non-leaf tensors, gradients are accumulated in the `grad` attribute only if the `_retain_grad` flag is set
          to True. This flag allows users to specify whether they want to retain gradients for non-leaf tensors after 
          backpropagation. By default, non-leaf tensors do not retain gradients to conserve memory.

        Additionally, the function updates `_accumulated_gradient_to_propagate`, which stores the gradient to be sent 
        backward to previous tensors in the computational graph. This temporary storage holds gradients until the tensor 
        has received all expected gradients, after which it will propagate them backward.
        """
        # Accumulate the gradient for the current tensor (only for leaf tensors or tensors marked to retain gradient)
        if self.is_leaf or self._retain_grad:
            if self.grad is None:
                self.grad = incoming_gradient
            else:
                self.grad += incoming_gradient

        # Accumulate the gradient to be propagated backward
        if self._accumulated_gradient_to_propagate is None:
            self._accumulated_gradient_to_propagate = incoming_gradient
        else:
            self._accumulated_gradient_to_propagate += incoming_gradient

    def _propagate_gradient(self, retain_graph: bool = False) -> None:
        """
        Propagate the accumulated gradient backward through the computational graph.

        This method sends the accumulated gradient to the `GradientFunction` that was responsible for creating this 
        tensor, allowing gradients to flow back through the network.

        The gradient is only propagated if the following conditions are met:
        1. The tensor has an associated `GradientFunction` (i.e., it is not a leaf node in the computational graph).
        2. The tensor has received all expected gradients (`_pending_gradients_count` is zero). This ensures that the 
        tensor waits until it has accumulated all gradients from the operations it was involved in.

        The `retain_graph` parameter determines whether the computational graph is preserved after the backward pass. 
        This is crucial for scenarios where multiple backward passes are necessary, such as in higher-order gradient 
        calculations. 

        After propagating the gradient, the accumulated gradient used for propagation is reset to `None`, preparing this 
        tensor for potential future backward passes. 
        
        The decision to release or retain the graph is taken at this stage because it ensures that the graph is released 
        only after all gradients have been fully propagated. It prevents unnecessary memory usage by releasing the graph 
        when it is no longer needed.
        """
        if self.grad_fn is not None and self._pending_gradients_count == 0:
            # Propagate the accumulated gradient backward to the GradientFunction that creates this tensor
            self.grad_fn.backward(self._accumulated_gradient_to_propagate, retain_graph)

            # Reset the accumulated gradient after propagation
            self._accumulated_gradient_to_propagate = None

            # Handle the retain_graph functionality
            if not retain_graph:
                self._release_graph()

    def _release_graph(self):
        """
        Release the computational graph.

        This method is called after the backward pass when retain_graph is False. It clears the reference to the 
        gradient function to free up memory and prevent further backward passes on the same graph. The method behaves 
        differently based on whether the tensor is a leaf or a non-leaf node in the computational graph:

        - Non-Leaf Tensors: For tensors that are results of an operation (non-leaf), the method clears their gradient 
        function (`grad_fn`). If `grad_fn` is already None, it indicates an attempt to backpropagate through an 
        already released graph, and thus a RuntimeError is raised.

        - Leaf Tensors: Leaf tensors are the initial tensors in the graph (like input data) and do not have a `grad_fn`. 
        Therefore, for leaf tensors, this method does not raise an error if `grad_fn` is None, as this is the expected 
        state for such tensors.
        """
        if self.grad_fn is not None:
            # Clear the gradient function to release the graph for non-leaf tensors 
            self.grad_fn = None

    def _propagate_reference_signal(self) -> None:
        """
        Propagate a reference signal through the computational graph to prepare for gradient backpropagation.

        This method is a preparatory step in the backward pass. Its primary function is to count the number of 
        operations each tensor in the graph is involved in. This count is crucial because it determines when a tensor is 
        ready to propagate its accumulated gradient backward. A tensor will only propagate its gradient back once it has 
        received all expected gradients from the operations it participated in. 

        By tracking this count, the method minimizes unnecessary traversals of the graph, ensuring that each tensor 
        waits until it has all its required gradient contributions before participating in the backward pass. This 
        optimization is significant in complex graphs where there are millions of tensors and operations.

        The method operates in two modes:
            - Initial call (when `_pending_gradients_count` is zero): This is typically the case when the backward pass 
            is initiated. The method will propagate the reference signal to the `GradientFunction` that created it, 
            which in turn propagates the signal to its input tensors.

            - Subsequent calls (when `_pending_gradients_count` is greater than zero): In this case, the tensor has 
            already received a part of its expected gradient. The method decrements the `_pending_gradients_count` 
            count, indicating that one less gradient is now expected.
        """
        # Start or continue the process of propagating the reference signal
        if self._pending_gradients_count == 0 and self.grad_fn is not None:
            # If this is the initial call, propagate the signal to the gradient function
            self.grad_fn._propagate_reference_signal()
        elif self._pending_gradients_count > 0:
            # If this is a subsequent call, decrement the backward reference count
            self._pending_gradients_count -= 1

    def _increment_pending_gradients(self) -> None:
        """
        Increment the count of pending gradients for this tensor and propagate this increment up the graph.

        This method is invoked when a tensor is identified as part of an operation in the computational graph that 
        contributes to the final output. By incrementing the `_pending_gradients_count` count, the tensor acknowledges 
        that it is expecting to receive a gradient from this operation during the backward pass.

        The incremented count serves two purposes:
        1. It keeps track of how many operations involve this tensor, indicating how many gradients this tensor should 
        expect to receive in the backward pass.
        2. It signals the tensor to wait until all its expected gradients are received before propagating its own gradient 
        backward, thereby ensuring the correct accumulation of gradients.

        After incrementing the reference count, this method propagates the reference signal further up the graph. This 
        propagation continues until it reaches tensors that are not part of any further operations, effectively preparing 
        the entire graph for efficient gradient propagation.

        If the tensor is a result of an operation (i.e., it has a gradient function), this method also triggers the 
        reference signal propagation in the gradient function, ensuring that all tensors upstream in the graph are 
        similarly prepared.
        """
        # Increment the count of pending gradients
        self._pending_gradients_count += 1

        # Propagate the count increment signal up to the gradient function, if it exists
        if self.grad_fn is not None:
            self.grad_fn._propagate_reference_signal()

    def zero_grad(self) -> Tensor:
        # Reset the gradient of the tensor and return the Tensor
        self.grad = None

        return self

    def retain_grad(self) -> None:
        # Allows to retain the gradient in non-leaf tensors
        self._retain_grad = True

    def _increment_version(self) -> None:
        """
        Increment the version counter of the tensor.

        This method should be called every time an in-place operation is performed on the tensor.
        """
        self._version += 1

    def detach(self) -> Tensor:
        """
        Create a new tensor that shares the same data but is not part of the computational graph. So no gradient will 
        be backpropagated along this variable.
        """
        return Tensor(self.buffer, requires_grad=False)

    def to_python(self) -> Union[float, int, bool, list]:
        # Convert the tensor's data to a Python scalar or nested list
        return self.buffer.to_python()

    def serialize_tensor(self) -> str:
        return f"Tensor(data={self.to_python()}, requires_grad={self.requires_grad})"

    @staticmethod
    def _serialized_pattern() -> str:
        # Define a regex pattern to match the serialized tensor format
        return r"Tensor\(data=(.+), requires_grad=(True|False)\)$"

    @staticmethod
    def validate_serialized_tensor(serialized_str: str) -> bool:
        # Use the re.match function to check if the entire string matches the pattern
        if re.match(Tensor._serialized_pattern(), serialized_str):
            return True  # The string is valid
        else:
            return False  # The string is invalid
    
    @staticmethod
    def deserialize_tensor(serialized_str: str) -> Tensor:
        match = re.match(Tensor._serialized_pattern(), serialized_str)
        if match:
            # Split the string in data and requires_grad strings
            data_str, requires_grad_str = match.groups()

            # Convert each element into Python values
            data = parse_value(data_str)
            requires_grad = requires_grad_str == "True"

            return Tensor(data=data, requires_grad=requires_grad)
        else:
            raise ValueError("Invalid serialized tensor format")


class GradientFunction():

    def __init__(self, operation: mlops.Operation = None, inputs: list[Tensor] = []) -> None:
        self.operation = operation
        self.inputs = inputs
        self.input_versions = {id(tensor): tensor._version for tensor in inputs}
        self._reference_signal_propagated = False

    def _propagate_reference_signal(self) -> None:
        """
        Propagate a reference signal through the inputs of this operation.

        This method increases the pending gradient count of each input tensor. It ensures that each tensor knows how 
        many gradients it should expect, which is crucial for the correct accumulation and backpropagation of gradients.

        The propagation occurs only once to prevent multiple counts for the same operation.
        """
        if not self._reference_signal_propagated:
            self._reference_signal_propagated = True
            for input_tensor in self.inputs:
                input_tensor._increment_pending_gradients()

    def backward(self, incoming_gradient: Buffer, retain_graph: bool = False) -> None:
        """
        Backpropagate the gradient through the operation.

        This method first checks the version of each input tensor to ensure they have not been modified since their
        last use in an operation. If any tensor's version has changed, it indicates that an in-place operation has
        been performed on the tensor after its involvement in the operation, making the current computational graph
        invalid for backpropagation. In such cases, a RuntimeError is raised.

        After validating tensor versions, the method computes the gradients for the operation and propagates them
        backward to each input tensor. The backward propagation continues recursively through the graph until reaching
        the leaf tensors.
        """
        if self.operation:
            # Check for version consistency of input tensors to ensure graph integrity
            for input_tensor in self.inputs:
                if input_tensor._version != self.input_versions[id(input_tensor)]:
                    raise RuntimeError(
                        f"The tensor {repr(input_tensor)} has been modified (version is {input_tensor._version}) since "
                        f"its use in this operation (when version was {self.input_versions[id(input_tensor)]}). "
                        "Backward pass is not allowed after in-place operations."
                    )

            # Computes the gradient of the operation
            gradients = self.operation.backward(incoming_gradient)

            # We expect a tuple with the gradients but some operations have only one so we wrap the gradient in a tuple
            if not isinstance(gradients, tuple):
                gradients = (gradients,)

            # Propagate the gradient of the operation to its input tensors
            for input_tensor, grad in zip(self.inputs, gradients):
                input_tensor.backward(grad, retain_graph)

            # Reset the reference signal flag for potential reuse of this graph
            self._reference_signal_propagated = False

    def __str__(self) -> str:
        inputs_str = ", ".join(f"<Tensor {hex(id(tensor))}>" for tensor in self.inputs)
        return f"<GradientFunction {self.operation} with inputs [{inputs_str}]>"


def apply_op(operation_cls: mlops.Operation, *tensors: Tensor, **kwargs) -> Tensor:
    # The output tensor requires gradient if any of the input tensors also require it
    needs_input_grad = [tensor.requires_grad for tensor in tensors]
    if any(needs_input_grad):
        requires_grad = True
    elif None in needs_input_grad:
        requires_grad = None
    else:
        requires_grad = False

    # We need an object of the operation to make the forward pass
    operation_object = operation_cls(needs_input_grad)

    # Do the forward pass with the buffers of the tensors
    buffer = operation_object.forward(*[tensor.buffer for tensor in tensors], **kwargs)

    # Save the buffer in a new tensor
    output_tensor = Tensor(buffer, requires_grad=requires_grad)

    # If the output tensor requires a gradient, set up the gradient function
    if output_tensor.requires_grad:
        output_tensor.grad_fn = GradientFunction(operation=operation_object, inputs=tensors)
        output_tensor.is_leaf = False  # The tensor is a result of an operation, hence not a leaf

    return output_tensor
