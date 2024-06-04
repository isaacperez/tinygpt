import math
from typing import Any

from tinygpt.tensor import Tensor
from tinygpt.module import Module
from tinygpt.utils import DType


class FullyConnectedLayer(Module):
    # A fully connected layer (or dense layer) which applies a linear transformation to the incoming data: `y = xW + b`

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weights = Tensor.uniform(shape=(input_dims, output_dims), low=-scale, high=scale, requires_grad=True)
        if bias:
            self.bias = Tensor.uniform(shape=(output_dims,), low=-scale, high=scale, requires_grad=True)

    def _extra_repr(self) -> str:
        return f"input_dims={self.weights.shape[0]}, output_dims={self.weights.shape[1]}, bias={'bias' in self}"

    def __call__(self, x: Tensor) -> Tensor:
        if "bias" in self:
            return x.dot(self.weights) + self.bias
        else:
            return x.dot(self.weights)


class MLP(Module):

    activation_functions = {
        'linear': lambda tensor: tensor,
        'relu': lambda tensor: tensor.relu()
    }

    def __init__(self, input_dims: int,  hidden_dims: list[int], activation_fn: str, bias: bool = True) -> None:
        super().__init__()

        if activation_fn not in self.activation_functions:
            raise ValueError(
                f"Unknown activation function '{activation_fn}'. "
                f"Expecting one of {list(self.activation_functions.keys())}"
            )

        self.activation_fn = activation_fn
        self.layers = []
        for out_features in hidden_dims:
            self.layers.append(FullyConnectedLayer(input_dims, out_features, bias))
            input_dims = out_features

    def _extra_repr(self) -> str:
        return f"activation_fn='{self.activation_fn}'"

    def __call__(self, x: Tensor) -> Tensor:
        # Apply a sequence of linear transformations to the incoming tensor interleaving the activation function
        output = x
        for layer in self.layers:
            output = MLP.activation_functions[self.activation_fn](layer(output))

        return output
    

class Embedding(Module):

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = Tensor.normal(shape=(self.num_embeddings, self.embedding_dim), requires_grad=True)

    def _extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"

    def __call__(self, indices: Tensor) -> Tensor:
        # Valdiate the indices
        if not isinstance(indices, Tensor):
            raise TypeError(f"Expecting a Tensor, but found {type(indices)}")
        
        if indices.dtype != DType.int32:
            raise ValueError("dtype of indices must be int32")
        
        if indices.ndim != 2:
            raise ValueError(f"Expecting a tensor with two dimensions, but found {indices.ndim} dimensions")
        
        # Extract each embedding based on the indices and concatenate them
        individual_embeddings = [
            self.embeddings[idx.to_python(), :].reshape((1, self.embedding_dim)) 
            for idx in indices.reshape((indices.shape[0] * indices.shape[1],))
        ]

        return Tensor.concatenate(individual_embeddings, axis=0).reshape((*indices.shape, self.embedding_dim))