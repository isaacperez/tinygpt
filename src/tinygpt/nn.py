from tinygpt.tensor import Tensor
from tinygpt.module import Module


class FullyConnectedLayer(Module):
    # A fully connected layer (or dense layer) which applies a linear transformation to the incoming data: `y = xW + b`

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        super().__init__()

        self.weights = Tensor.uniform(shape=(input_dims, output_dims), requires_grad=True)
        if bias:
            self.bias = Tensor.uniform(shape=(output_dims,), requires_grad=True)

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