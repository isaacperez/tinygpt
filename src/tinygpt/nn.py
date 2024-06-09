import math
from typing import Any, Union

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
    

class LayerNorm(Module):

    def __init__(
            self, 
            normalized_shape: Union[tuple[int, ...], int], 
            eps: float = 1e-05, 
            elementwise_affine: bool = True, 
            bias: bool = True
        ) -> None:
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        if len(normalized_shape) < 1:
            raise ValueError(
                f"Expected normalized_shape to be at least 1-dimensional, i.e., "
                f"containing at least one element, but got normalized_shape = {normalized_shape}"
            )
        
        if any(value < 1 for value in normalized_shape):
            raise RuntimeError(f"Expected positive values for normalized_shape, but found {normalized_shape}")

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.bias = bias
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weights = Tensor.ones(shape=self.normalized_shape, requires_grad=True)
            self.bias = Tensor.zeros(shape=self.normalized_shape, requires_grad=True)

    def _extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"

    def __call__(self, tensor: Tensor) -> Tensor:        
        # Validate the input tensor
        if tensor.dtype != DType.float32:
            raise ValueError(f"Expecting input tensor with dtype float32, but found {tensor.dtype}")
        
        if tensor.ndim < 2:
            raise ValueError(f"Expecting input with two dimensions, but found {tensor.ndim} dimensions")
        
        if self.elementwise_affine and tensor.shape[-len(self.normalized_shape):] != self.normalized_shape:
            raise RuntimeError(
                f"Given normalized_shape {self.normalized_shape}, expected input with shape "
                f"{tuple(['*'] + list(self.normalized_shape))}, but got input with shape {tensor.shape}"
            )

        # Calculate the mean and the variance of the input tensor to apply the normalization
        axes = tuple(tensor.ndim - (i + 1) for i in range(len(self.normalized_shape)))
       
        mean = tensor.mean(axes=axes, keepdim=True)
        mean_square = (tensor ** 2.0).mean(axes=axes, keepdim=True)

        var = mean_square - mean ** 2.0

        # Normalize the input tensor
        tensor = (tensor - mean) / ((var + self.eps) ** 0.5)

        # Apply affine transformation
        if self.elementwise_affine:
            tensor = tensor * self.weights + self.bias
        
        return tensor

    