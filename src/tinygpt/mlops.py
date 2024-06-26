from typing import Union, Any

from tinygpt.buffer import Buffer
from tinygpt.utils import argsort


class Operation():

    def __init__(self, needs_input_grad: list) -> None:
        self.needs_input_grad = needs_input_grad

    def forward(self, *args, **kwargs) -> Buffer:
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self, *args, **kwargs) -> Any:
        raise RuntimeError(f"backward not implemented for {type(self)}")

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} {hex(id(self))}>"


class Add(Operation):

    def forward(self, first_buffer: Buffer, second_buffer: Buffer) -> Buffer:
        return first_buffer + second_buffer

    def backward(self, incoming_grad: Buffer) -> tuple[Union[Buffer, None], Union[Buffer, None]]:
        grad_first = incoming_grad if self.needs_input_grad[0] else None
        grad_second = incoming_grad if self.needs_input_grad[1] else None

        return grad_first, grad_second


class Sub(Operation):

    def forward(self, first_buffer: Buffer, second_buffer: Buffer) -> Buffer:
        return first_buffer - second_buffer

    def backward(self, incoming_grad: Buffer) -> tuple[Union[Buffer, None], Union[Buffer, None]]:
        grad_first = incoming_grad if self.needs_input_grad[0] else None
        grad_second = -incoming_grad if self.needs_input_grad[1] else None

        return grad_first, grad_second


class Neg(Operation):

    def forward(self, buffer: Buffer) -> Buffer:
        return -buffer

    def backward(self, incoming_grad: Buffer) -> Union[Buffer, None]:
        return -incoming_grad if self.needs_input_grad[0] else None


class Mul(Operation):

    def forward(self, first_buffer: Buffer, second_buffer: Buffer) -> Buffer:
        self.first_buffer, self.second_buffer = first_buffer, second_buffer
        return first_buffer * second_buffer

    def backward(self, incoming_grad: Buffer) -> tuple[Union[Buffer, None], Union[Buffer, None]]:
        grad_first = self.second_buffer * incoming_grad if self.needs_input_grad[0] else None
        grad_second = self.first_buffer * incoming_grad if self.needs_input_grad[1] else None

        return grad_first, grad_second


class Div(Operation):

    def forward(self, first_buffer: Buffer, second_buffer: Buffer) -> Buffer:
        self.first_buffer, self.second_buffer = first_buffer, second_buffer
        return first_buffer / second_buffer

    def backward(self, incoming_grad: Buffer) -> tuple[Union[Buffer, None], Union[Buffer, None]]:
        grad_first = incoming_grad / self.second_buffer if self.needs_input_grad[0] else None

        grad_second = None
        if self.needs_input_grad[1]:
            grad_second = - ((self.first_buffer * incoming_grad) / (self.second_buffer * self.second_buffer))

        return grad_first, grad_second


class Pow(Operation):

    def forward(self, buffer: Buffer, exponent: Union[int, float]) -> Buffer:
        self.buffer, self.exponent = buffer, exponent
        return buffer ** exponent

    def backward(self, incoming_grad: Buffer) -> Union[Buffer, None]:
        if self.needs_input_grad[0]:
            return (self.exponent * self.buffer ** (self.exponent - 1)) * incoming_grad
        else:
            return None


class Log(Operation):

    def forward(self, buffer: Buffer) -> Buffer:
        self.buffer = buffer
        return buffer.log()

    def backward(self, incoming_grad: Buffer) -> Union[Buffer, None]:
        return incoming_grad / self.buffer if self.needs_input_grad[0] else None


class Exp(Operation):

    def forward(self, buffer: Buffer) -> Buffer:
        self.result = buffer.exp()
        return self.result

    def backward(self, incoming_grad: Buffer) -> Union[Buffer, None]:
        return self.result * incoming_grad if self.needs_input_grad[0] else None


class Maximum(Operation):

    def forward(self, first_buffer: Buffer, second_buffer: Buffer) -> Buffer:
        self.first_buffer, self.second_buffer = first_buffer, second_buffer
        return first_buffer.maximum(second_buffer)

    def backward(self, incoming_grad: Buffer) -> tuple[Union[Buffer, None], Union[Buffer, None]]:
        first_grad = None
        if self.needs_input_grad[0]:
            first_grad = (self.first_buffer > self.second_buffer).float() * incoming_grad

        second_grad = None
        if self.needs_input_grad[1]:
            second_grad = (self.second_buffer > self.first_buffer).float() * incoming_grad

        return first_grad, second_grad


class Relu(Operation):

    def forward(self, buffer: Buffer) -> Buffer:
        self.buffer = buffer
        return buffer.maximum(0.)

    def backward(self, incoming_grad: Buffer) -> Union[Buffer, None]:
        return (self.buffer > 0.).float() * incoming_grad if self.needs_input_grad[0] else None


class Sum(Operation):

    def forward(self, buffer: Buffer, axes: tuple) -> Buffer:
        self.input_shape = buffer.shape
        return buffer.sum(axes)

    def backward(self, incoming_grad: Buffer) -> Buffer:
        return incoming_grad.expand(self.input_shape) if self.needs_input_grad[0] else None


class Max(Operation):

    def forward(self, buffer: Buffer, axes: tuple) -> Buffer:
        self.buffer, self.axes = buffer, axes
        self.result = buffer.max(axes)
        return self.result

    def backward(self, incoming_grad: Buffer) -> Buffer:
        if self.needs_input_grad[0]:
            max_is_1s = 1.0 - (self.buffer < self.result.expand(self.buffer.shape)).float()
            div = max_is_1s.sum(self.axes).expand(self.buffer.shape)
            return (max_is_1s / div) * incoming_grad.expand(self.buffer.shape)
        else:
            return None


class Reshape(Operation):

    def forward(self, buffer: Buffer, new_shape: tuple) -> Buffer:
        self.input_shape = buffer.shape
        return buffer.reshape(new_shape)

    def backward(self, incoming_grad: Buffer) -> Union[Buffer, None]:
        return incoming_grad.reshape(self.input_shape) if self.needs_input_grad[0] else None


class Expand(Operation):

    def forward(self, buffer: Buffer, new_shape: tuple) -> Buffer:
        self.input_shape = buffer.shape
        return buffer.expand(new_shape)

    def backward(self, incoming_grad: Buffer) -> Union[Buffer, None]:
        if self.needs_input_grad[0]:
            axes = tuple(idx for idx, val in enumerate(incoming_grad.shape) if self.input_shape[idx] != val)
            return incoming_grad.sum(axes)
        else:
            return None


class Permute(Operation):

    def forward(self, buffer: Buffer, dims: tuple) -> Buffer:
        self.input_order = dims
        return buffer.permute(dims)

    def backward(self, incoming_grad: Buffer) -> Union[Buffer, None]:
        return incoming_grad.permute(argsort(self.input_order)) if self.needs_input_grad[0] else None
    

class Slice(Operation):

    def forward(self, buffer: Buffer, index: Union[int, slice, tuple]) -> Buffer:
        self.input_shape = buffer.shape
        self.input_index = index
        return buffer[index]

    def backward(self, incoming_grad: Buffer) -> Union[Buffer, None]:
        if self.needs_input_grad[0]:
            # Only the places where we have indexed the input tensor have gradient
            output_grad = Buffer.zeros(self.input_shape)
            
            # Place the gradient in the correct location
            output_grad[self.input_index] = incoming_grad

            return output_grad
        else:
            return None


class Concatenate(Operation):

    def forward(self, *buffers: tuple[Buffer], axis: int) -> Buffer:
        self.buffers = buffers
        self.axis = axis
        return Buffer.concatenate(buffers, axis=axis)

    def backward(self, incoming_grad: Buffer) -> tuple[Union[Buffer, None]]:
        splits = []
        start_idx = 0
        for i, buffer in enumerate(self.buffers):
            if self.needs_input_grad[i]:
                # Extract the slice of the incoming gradient that corresponds to the current buffer
                end_idx = start_idx + buffer.shape[self.axis]
                slices = [slice(None)] * buffer.ndim
                slices[self.axis] = slice(start_idx, end_idx)

                split_grad = incoming_grad[tuple(slices)]

                splits.append(split_grad)
                start_idx = end_idx

            else:
                splits.append(None)

        return tuple(splits)
    

class Tril(Operation):

    def forward(self, buffer: Buffer, diagonal: int = 0) -> Buffer:
        self.diagonal = diagonal
        return buffer.tril(diagonal)

    def backward(self, incoming_grad: Buffer) -> Union[Buffer, None]:
        # The backward method for tril will only pass the gradient of the lower triangular part
        # Similar to the forward pass, zero out the upper triangular part of the incoming gradient
        if self.needs_input_grad[0]:
            tril_grad = incoming_grad.tril(self.diagonal)
            return tril_grad
        else:
            return None