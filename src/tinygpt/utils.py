from __future__ import annotations
from typing import Any
from enum import Enum


class DType(Enum):
    # Each type has an equivalent Python type and a priority associated
    float32 = float
    int32 = int
    bool = bool

    def cast(self, input_value: Any) -> Any:
        return self.value(input_value)

    @staticmethod
    def deduce_dtype(value: Any) -> DType:
        for dtype in DType:
            if type(value) is dtype.value:
                return dtype
        raise RuntimeError(f"Could not infer dtype of type {type(value)}")
