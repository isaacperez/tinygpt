from typing import Union, Any

from tinygpt.buffer import Buffer


class Operation():

    def __init__(self, needs_input_grad: list) -> None:
        self.needs_input_grad = needs_input_grad

    def forward(self, *args, **kwargs) -> Buffer:
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self, *args, **kwargs) -> Any:
        raise RuntimeError(f"backward not implemented for {type(self)}")

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} {hex(id(self))}>"


class Sum(Operation):

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

    def backward(self, incoming_grad: Buffer) -> tuple[Union[Buffer, None]]:
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
        return first_buffer * second_buffer

    def backward(self, incoming_grad: Buffer) -> tuple[Union[Buffer, None], Union[Buffer, None]]:
        grad_first = incoming_grad / self.second_buffer if self.needs_input_grad[0] else None

        grad_second = None
        if self.needs_input_grad[1]:
            grad_second = - ((self.first_buffer * incoming_grad) / (self.second_buffer * self.second_buffer))

        return grad_first, grad_second
