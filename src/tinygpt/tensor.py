from __future__ import annotations
from typing import Any

from tinygpt.utils import DType
from tinygpt.buffer import Buffer


class Tensor():

    def __init__(
            self,
            data: Any,
            dtype: DType = None,
            requires_grad=False
    ) -> None:
        # Save the data in a buffer
        self.buffer = Buffer(data, dtype)

        # Gradient-related metadata
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    def __repr__(self) -> str:
        if self.ndim > 1 and any(value == 0 for value in self.shape):
            return f"<Tensor {self.buffer}, size={self.shape}>"
        else:
            return f"<Tensor {self.buffer}>"

    @property
    def shape(self):
        return self.buffer.shape

    @property
    def ndim(self):
        return self.buffer.ndim

    @property
    def dtype(self):
        return self.buffer.dtype
