from __future__ import annotations
from typing import Any

from tinygpt.utils import DType
from tinygpt.buffer import Buffer
import tinygpt.mlops as mlops


class Tensor():

    def __init__(self, data: Any, dtype: DType = None, requires_grad=False) -> None:
        # Save the data in a buffer
        self.buffer = Buffer(data, dtype)

        # Gradient-related metadata
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self._backward_references = 0

        # Only float tensors can require gradients
        if self.requires_grad and self.dtype != DType.float32:
            raise RuntimeError("Only Tensors of floating point dtype can require gradients")

    def __repr__(self) -> str:
        return (
            f"<Tensor {hex(id(self))}: {self.buffer}, shape={self.shape}, dtype={self.dtype}, "
            f"requires_grad={self.requires_grad}>"
        )

    def __str__(self) -> str:
        return f"<Tensor {self.buffer}, shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad}>"

    @property
    def shape(self):
        return self.buffer.shape

    @property
    def ndim(self):
        return self.buffer.ndim

    @property
    def dtype(self):
        return self.buffer.dtype

    def __add__(self, other: Any) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return apply_op(mlops.Sum, self, other)

    def _increment_backward_references(self) -> None:
        if self.requires_grad:
            self._backward_references += 1

    def backward(self, incoming_gradient=None) -> None:
        if self.requires_grad:
            # Decrease the counter of received gradients
            self._backward_references -= 1

            # Checks if any gradient is coming (first call to backward)
            if incoming_gradient is None:
                # Backward can only be called the first time from scalar tensors
                if self.ndim != 0:
                    raise RuntimeError(f"backward can only be called for scalar tensors. Found shape {self.shape}")

                incoming_gradient = Buffer(1.0)

            # Add incoming gradient to your local gradient
            if self.grad is None:
                self.grad = incoming_gradient
            else:
                self.grad += incoming_gradient

            # Propagate your gradient to the function that created you when you have received all the expected gradients
            if self._backward_references == 0:
                if self.grad_fn is not None:
                    self.grad_fn.backward(self.grad)

                # Graph cleanup after calling backward
                self.grad_fn = None


class GradientFunction():

    def __init__(self, operation: mlops.Operation = None, inputs: list[Tensor] = []) -> None:
        self.operation = operation
        self.inputs = inputs

    def backward(self, incoming_gradient: Tensor) -> None:
        if self.operation is not None:
            # Calculate the gradient of the operation with the incoming gradient
            gradients = self.operation.backward(incoming_gradient)

            # Propagate the gradient of the operation to its input tensors
            for input_tensor, grad in zip(self.inputs, gradients):
                input_tensor.backward(grad)

    def __str__(self) -> str:
        # Create a simplified representation of the input tensors
        inputs_repr = []
        for input_tensor in self.inputs:
            inputs_repr.append(f"<Tensor {hex(id(input_tensor))}>")

        inputs_repr = ", ".join(inputs_repr)

        return f"<GradientFunction {self.operation} with {len(self.inputs)} inputs ({inputs_repr})>"


def apply_op(operation_cls: mlops.Operation, *tensors: Tensor, **kwargs) -> Tensor:
    # We need to know which input tensor requires gradient
    needs_input_grad = [tensor.requires_grad for tensor in tensors]

    # The output tensor requires gradient if any of the input tensors also require it
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

    # We update the GradientFunction object of the output tensor when it requires gradient. Specifically, we store
    # the operation that created the output tensor and the input tensors used in that operation
    if output_tensor.requires_grad:
        output_tensor.grad_fn = GradientFunction(operation=operation_object, inputs=tensors)

        # Increment backward reference count for each input tensor that requires gradient
        for tensor in tensors:
            if tensor.requires_grad:
                tensor._increment_backward_references()

    return output_tensor
