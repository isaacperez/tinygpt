from tinygpt.buffer import Buffer


class Operation():

    def __init__(self, needs_input_grad: list):
        self.needs_input_grad = needs_input_grad

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self, *args, **kwargs):
        raise RuntimeError(f"backward not implemented for {type(self)}")

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} {hex(id(self))}>"


class Sum(Operation):

    def __init__(self, needs_input_grad: list):
        super().__init__(needs_input_grad)

    def forward(self, first_buffer: Buffer, second_buffer: Buffer) -> Buffer:
        return first_buffer + second_buffer

    def backward(self, incoming_grad: Buffer) -> Buffer:
        return incoming_grad if self.needs_input_grad[0] else None, incoming_grad if self.needs_input_grad[1] else None
