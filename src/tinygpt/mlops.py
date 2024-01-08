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
        axes = tuple(idx for idx, val in enumerate(incoming_grad.shape) if self.input_shape[idx] != val)
        return incoming_grad.sum(axes) if self.needs_input_grad[0] else None


class Permute(Operation):

    def forward(self, buffer: Buffer, dims: tuple) -> Buffer:
        self.input_order = dims
        return buffer.permute(dims)

    def backward(self, incoming_grad: Buffer) -> Union[Buffer, None]:
        return incoming_grad.permute(argsort(self.input_order))
