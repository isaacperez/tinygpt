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


def print_dag(node: Any, indent='', last=True, is_grad_fn=False) -> None:
    # Recursively prints the DAG of tensors and their gradient functions

    # Define the tree structure's branch elements
    tree_elements = '└──' if last else '├──'

    # Print the current node
    print(f"{indent}{tree_elements} {node if is_grad_fn else repr(node)}")

    # Prepare the next level of indentation
    indent += '   ' if last else '│  '

    # Identify the next set of nodes to visit
    next_nodes = []
    if is_grad_fn:
        next_nodes = node.inputs
    elif node.grad_fn:
        next_nodes = [node.grad_fn]

    # Recursively visit each next node
    for i, child in enumerate(next_nodes):
        if child is not None:
            print_dag(child, indent, i == len(next_nodes) - 1, not is_grad_fn)


def argsort(x):
    return type(x)(sorted(range(len(x)), key=x.__getitem__))
