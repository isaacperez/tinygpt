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


class CasualSelfAttention(Module):

    def __init__(self, embedding_dim: int, max_seq_length: int, num_heads: int) -> None:
        super().__init__()

        # Validate the parameters
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding dimension must be divisible by num_heads")
        
        if num_heads > embedding_dim:
            raise ValueError("num_heads is greater than the embedding dimension")
        
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.num_heads = num_heads

        self.head_dim = embedding_dim // num_heads
        self.scale_factor = 1 / math.sqrt(self.embedding_dim)

        self.query = FullyConnectedLayer(self.embedding_dim, self.embedding_dim, bias=False)
        self.key = FullyConnectedLayer(self.embedding_dim, self.embedding_dim, bias=False)
        self.value = FullyConnectedLayer(self.embedding_dim, self.embedding_dim, bias=False)

        self.out = FullyConnectedLayer(self.embedding_dim, self.embedding_dim, bias=False)

        # Prepare the casual mask and the expected value for the zero positions
        self.casual_mask = Tensor.ones((1, 1, self.max_seq_length, self.max_seq_length)).tril()
        self.casual_value = Tensor(float("-inf"), dtype=DType.float32).reshape(
            (1, 1, 1, 1)
        ).expand((1, 1, self.max_seq_length, self.max_seq_length)).tril(diagonal=-1).transpose(3, 2)

    def _extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, max_seq_length={self.max_seq_length}, num_heads={self.num_heads}"
        )
    
    def __call__(self, tensor: Tensor) -> Tensor:  
        
        # Only 3-dimensional tensors are valid
        if len(tensor.shape) != 3:
            raise RuntimeError(f"Expecting a 3-dimensional tensor but found tensor with shape {tensor.shape}")

        batch_size, seq_length, embed_dim = tensor.shape

        # Ensure the last dimension matches embedding_dim
        if embed_dim != self.embedding_dim:
            raise RuntimeError(f"Embedding dimension of the input ({embed_dim}) does not match {self.embedding_dim}")

        if seq_length > self.max_seq_length:
            raise RuntimeError(
                f"Sequence dimension of the input ({seq_length}) greater than max allowed ({self.max_seq_length})")
        
        # Project to query, key and value
        query_proj = self.query(tensor)
        key_proj = self.key(tensor)
        value_proj = self.value(tensor)

        # Split the embedding dimension into chuncks for each head: [BS, S, E] -> [BS, S, NH, HD] -> [B, NH, S, HE]
        query_proj = query_proj.reshape((batch_size, seq_length, self.num_heads, self.head_dim)).transpose(1, 2) 
        key_proj = key_proj.reshape((batch_size, seq_length, self.num_heads, self.head_dim)).transpose(1, 2) 
        value_proj = value_proj.reshape((batch_size, seq_length, self.num_heads, self.head_dim)).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = query_proj.dot(key_proj.transpose(2, 3)) * self.scale_factor
        
        # Apply the mask to the attention scores and set to -inf the zero positions
        attention_scores = attention_scores * self.casual_mask[:, :, :seq_length, :seq_length]
        attention_scores = attention_scores + self.casual_value[:, :, :seq_length, :seq_length]

        # Softmax to get the attention weights
        attention_weights = attention_scores.softmax(axis=3)

        # Multiply the attention weights with the values
        attention_output = attention_weights.dot(value_proj)

        # Concatenate all the head embeddings
        attention_output = attention_output.transpose(1, 2).reshape((batch_size, seq_length, embed_dim))

        # Project the output back to the original embedding dimension
        output = self.out(attention_output)

        return output