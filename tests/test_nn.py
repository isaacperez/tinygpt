import pytest

from tinygpt.tensor import Tensor
from tinygpt.buffer import Buffer
from tinygpt.nn import FullyConnectedLayer, MLP


def test_FullyConnectedLayer():
    # Wrong data types
    with pytest.raises(TypeError):
        _ = FullyConnectedLayer(in_features=None, out_features=12, use_bias=True)
    with pytest.raises(TypeError):
        _ = FullyConnectedLayer(in_features=12, out_features=None, use_bias=True)
    with pytest.raises(TypeError):
        _ = FullyConnectedLayer(in_features=12, out_features=12, use_bias=None)
    
    # Wrong values
    for wrong_value in [-1, 0]:
        with pytest.raises(ValueError):
            _ = FullyConnectedLayer(in_features=wrong_value, out_features=12, use_bias=True)
        
        with pytest.raises(ValueError):
            _ = FullyConnectedLayer(in_features=12, out_features=wrong_value, use_bias=True)

    # Do the inference
    for in_features, out_features in [(12, 12), (12, 6), (12, 24)]:
        
        layer_without_bias = FullyConnectedLayer(in_features=in_features, out_features=out_features, use_bias=False)
        layer_with_bias = FullyConnectedLayer(in_features=in_features, out_features=out_features, use_bias=True)

        # Check the modules have the expected components
        assert len([c for c in layer_without_bias.children()]) == 0
        assert len([c for c in layer_with_bias.children()]) == 0
        
        assert len([(n, c) for n, c in layer_without_bias.named_children()]) == 0
        assert len([(n, c) for n, c in layer_with_bias.named_children()]) == 0

        assert len([m for m in layer_without_bias.modules()]) == 1
        assert len([m for m in layer_with_bias.modules()]) == 1

        assert len([(n, m) for n, m in layer_without_bias.named_modules()]) == 1
        assert len([(n, m) for n, m in layer_with_bias.named_modules()]) == 1
        
        assert len([p for p in layer_without_bias.parameters()]) == 1
        assert len([p for p in layer_with_bias.parameters()]) == 2
        
        assert len([(n, p) for n, p in layer_without_bias.named_parameters()]) == 1
        assert len([(n, p) for n, p in layer_with_bias.named_parameters()]) == 2
        
        # Check parameters has the expected name and shape
        for n, p in layer_with_bias.named_parameters():
            assert n in ('weights', 'bias')

            if n == 'weights':
                assert p.shape == (in_features, out_features)
            else:
                assert p.shape == (out_features,)

        # Use different shapes for the input tensor
        for input_shape in [(in_features,), (24, in_features), (12, 3, in_features)]:
            input_tensor = Tensor.uniform(input_shape, requires_grad=True)

            # Do inference
            output_tensor_without_bias = layer_without_bias(input_tensor)
            output_tensor_with_bias = layer_with_bias(input_tensor)

            # Check the output
            assert output_tensor_without_bias.shape == (*input_shape[:-1], out_features)
            assert output_tensor_with_bias.shape == (*input_shape[:-1], out_features)

            assert output_tensor_without_bias.requires_grad
            assert output_tensor_with_bias.requires_grad

            # Do inference with a wrong shape for the input
            with pytest.raises(RuntimeError):
                _ = layer_without_bias(Tensor.uniform((*input_shape[:-1], input_shape[-1] + 1)))

            with pytest.raises(RuntimeError):
                _ = layer_with_bias((Tensor.uniform((*input_shape[:-1], input_shape[-1] + 1))))


def test_FullyConnectedLayer_zero_grad():
    pass


def test_FullyConnectedLayer_save_and_load():
    pass


def test_MLP():
    pass


def test_MLP_zero_grad():
    pass


def test_MLP_save_and_load():
    pass