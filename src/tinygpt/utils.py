from __future__ import annotations
from typing import Any
from enum import Enum


class DType(Enum):
    float32 = float
    int32 = int
    bool = bool

    def cast(self, input_value: Any) -> Any:
        return self.value(input_value)

    @staticmethod
    def prefered_order():
        return (DType.float32, DType.int32, DType.bool)

    @staticmethod
    def deduce_dtype(value: Any) -> DType:
        for dtype in DType:
            if type(value) is dtype.value:
                return dtype
        raise RuntimeError(f"Could not infer dtype of type {type(value)}")

    @staticmethod
    def type_promotion(first_type: DType, second_type: DType):
        for dtype in DType.prefered_order():
            if first_type == dtype or second_type == dtype:
                return dtype
        raise RuntimeError(f"first_type {type(first_type)} and second_type {type(second_type)} are not DType objects")


class MultidimensionalIndexGenerator:
    """Iterator to generate all possible index tuples for an array with a given shape"""

    def __init__(self, shape: tuple):
        self.shape = shape
        self.num_dimensions = len(shape)
        self.indices = [0] * self.num_dimensions
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration

        if self.num_dimensions == 0:
            self.finished = True
            return (0,)

        # Get the index for the current iteration
        result = tuple(self.indices)

        # Update indices for the next iteration
        for i in range(self.num_dimensions - 1, -1, -1):
            if self.indices[i] < self.shape[i] - 1:
                self.indices[i] += 1
                break
            else:
                self.indices[i] = 0
                if i == 0:
                    self.finished = True

        return result
