import pytest

from tinygpt.tensor import Tensor
from tinygpt.buffer import Buffer
from tinygpt.nn import FullyConnectedLayer, MLP


def test_FullyConnectedLayer():
    # Wrong data types
    with pytest.raises(TypeError):
        _ = FullyConnectedLayer(input_dims=None, output_dims=12, bias=True)
    with pytest.raises(TypeError):
        _ = FullyConnectedLayer(input_dims=12, output_dims=None, bias=True)
    
    # Wrong values
    for wrong_value in [-1, 0]:
        with pytest.raises(ValueError):
            _ = FullyConnectedLayer(input_dims=wrong_value, output_dims=12, bias=True)
        
        with pytest.raises(ValueError):
            _ = FullyConnectedLayer(input_dims=12, output_dims=wrong_value, bias=True)

    # Do the inference
    for input_dims, output_dims in [(12, 12), (12, 6), (12, 24)]:
        
        layer_without_bias = FullyConnectedLayer(input_dims=input_dims, output_dims=output_dims, bias=False)
        layer_with_bias = FullyConnectedLayer(input_dims=input_dims, output_dims=output_dims, bias=True)

        # Check the modules have the expected components
        assert len(layer_without_bias.children()) == 0
        assert len(layer_with_bias.children()) == 0

        assert len(layer_without_bias.modules()) == 1
        assert len(layer_with_bias.modules()) == 1

        assert len(layer_without_bias.named_modules()) == 1
        assert len(layer_with_bias.named_modules()) == 1
        
        assert len(layer_without_bias.parameters()) == 1
        assert len(layer_with_bias.parameters()) == 2
        
        assert len(layer_without_bias.trainable_parameters()) == 1
        assert len(layer_with_bias.trainable_parameters()) == 2
        
        assert len(layer_without_bias.leaf_modules()) == 0
        assert len(layer_with_bias.leaf_modules()) == 0

        # Check parameters has the expected name and shape
        for n, p in layer_with_bias.parameters().items():
            assert n in ('weights', 'bias')

            if n == 'weights':
                assert p.shape == (input_dims, output_dims)
            else:
                assert p.shape == (output_dims,)
            
            assert p.requires_grad

        # Use different shapes for the input tensor
        for input_shape in [(input_dims,), (24, input_dims), (12, 3, input_dims)]:
            input_tensor = Tensor.uniform(input_shape, requires_grad=True)

            # Do inference
            output_tensor_without_bias = layer_without_bias(input_tensor)
            output_tensor_with_bias = layer_with_bias(input_tensor)

            # Check the output
            assert output_tensor_without_bias.shape == (*input_shape[:-1], output_dims)
            assert output_tensor_with_bias.shape == (*input_shape[:-1], output_dims)

            assert output_tensor_without_bias.requires_grad
            assert output_tensor_with_bias.requires_grad

            # Do inference with a wrong shape for the input
            with pytest.raises(RuntimeError):
                _ = layer_without_bias(Tensor.uniform((*input_shape[:-1], input_shape[-1] + 1)))

            with pytest.raises(RuntimeError):
                _ = layer_with_bias((Tensor.uniform((*input_shape[:-1], input_shape[-1] + 1))))


def test_FullyConnectedLayer_zero_grad():
    # Call zero_grad after backward pass
    for input_dims, output_dims in [(12, 12), (12, 6), (12, 24)]:
        # Create the layers with different configurations
        layer_without_bias = FullyConnectedLayer(input_dims=input_dims, output_dims=output_dims, bias=False)
        layer_with_bias = FullyConnectedLayer(input_dims=input_dims, output_dims=output_dims, bias=True)

        # Do the forward and backward pass
        for input_shape in [(input_dims,), (24, input_dims), (12, 3, input_dims)]:
            input_tensor = Tensor.uniform(input_shape, requires_grad=True)

            # Do the forward pass
            output_tensor_without_bias = layer_without_bias(input_tensor)
            output_tensor_with_bias = layer_with_bias(input_tensor)

            # Do the backward pass
            output_tensor_without_bias.sum(axes=tuple(i for i in range(output_tensor_without_bias.ndim))).backward()
            output_tensor_with_bias.sum(axes=tuple(i for i in range(output_tensor_with_bias.ndim))).backward()

            # Check weights of the layer has gradient
            for p in layer_without_bias.parameters().values():
                assert p.grad is not None
                assert p.grad.shape == p.shape

            for p in layer_with_bias.parameters().values():
                assert p.grad is not None
                assert p.grad.shape == p.shape

            # Call zero_grad()
            layer_without_bias.zero_grad()
            layer_with_bias.zero_grad()

            # Check weights of the layer has gradient
            for p in layer_without_bias.parameters().values():
                assert p.grad is None

            for p in layer_with_bias.parameters().values():
                assert p.grad is None


def test_FullyConnectedLayer_freeze_and_unfreeze():
    for input_dims, output_dims in [(12, 12), (12, 6), (12, 24)]:
        # Create the layers with different configurations
        layer_without_bias = FullyConnectedLayer(input_dims=input_dims, output_dims=output_dims, bias=False)
        layer_with_bias = FullyConnectedLayer(input_dims=input_dims, output_dims=output_dims, bias=True)

        # Freeze the weights
        layer_without_bias.freeze()
        layer_with_bias.freeze()

        # Check weights of the layer doesn't requires gradient
        assert len(layer_without_bias.trainable_parameters()) == 0
        assert len(layer_with_bias.trainable_parameters()) == 0

        # Unfreeze the weights
        layer_without_bias.unfreeze()
        layer_with_bias.unfreeze()

        # Check weights of the layer doesn't requires gradient
        assert len(layer_without_bias.trainable_parameters()) == 1
        assert len(layer_with_bias.trainable_parameters()) == 2


def test_FullyConnectedLayer_train_and_eval():
    for input_dims, output_dims in [(12, 12), (12, 6), (12, 24)]:
        # Create the layers with different configurations
        layer_without_bias = FullyConnectedLayer(input_dims=input_dims, output_dims=output_dims, bias=False)
        layer_with_bias = FullyConnectedLayer(input_dims=input_dims, output_dims=output_dims, bias=True)

        # Freeze the weights
        layer_without_bias.eval()
        layer_with_bias.eval()

        # Check layers are not in training mode
        assert not layer_without_bias.training
        assert not layer_with_bias.training

        # Unfreeze the weights
        layer_without_bias.train()
        layer_with_bias.train()

        # Check layers are in training mode
        assert layer_without_bias.training
        assert layer_with_bias.training


def test_FullyConnectedLayer_save_and_load(tmp_path):
    # Create different layers and save and restore it
    tmp_dir = tmp_path / "test_FullyConnectedLayer_save_and_load"
    tmp_dir.mkdir()
    for input_dims, output_dims in [(12, 12), (12, 6), (12, 24)]:
        # Create the layers with different configurations
        layer_without_bias = FullyConnectedLayer(input_dims=input_dims, output_dims=output_dims, bias=False)
        layer_with_bias = FullyConnectedLayer(input_dims=input_dims, output_dims=output_dims, bias=True)

        # Do the forward and backward pass
        input_shape = (12, 3, input_dims)
        input_tensor = Tensor.uniform(input_shape, requires_grad=True)

        # Do the forward pass
        output_tensor_without_bias = layer_without_bias(input_tensor)
        output_tensor_with_bias = layer_with_bias(input_tensor)

        # Save the layers
        layer_without_bias.save_weights(str(tmp_dir / "layer_without_bias.json"))
        layer_with_bias.save_weights(str(tmp_dir / "layer_with_bias.json"))

        # Create the layer again
        layer_without_bias = FullyConnectedLayer(input_dims=input_dims, output_dims=output_dims, bias=False)
        layer_with_bias = FullyConnectedLayer(input_dims=input_dims, output_dims=output_dims, bias=True)

        # Load the weights
        layer_without_bias.load_weights(str(tmp_dir / "layer_without_bias.json"))
        layer_with_bias.load_weights(str(tmp_dir / "layer_with_bias.json"))

        # Do the inference again
        new_output_tensor_without_bias = layer_without_bias(input_tensor)
        new_output_tensor_with_bias = layer_with_bias(input_tensor)

        # Check results are equal to the results before
        assert all(new_output_tensor_without_bias.buffer == output_tensor_without_bias.buffer)
        assert all(new_output_tensor_with_bias.buffer == output_tensor_with_bias.buffer)

        # Try to load wrong weight file
        with pytest.raises(ValueError):    
            layer_without_bias.load_weights(str(tmp_dir / "layer_with_bias.json"))
        
        with pytest.raises(ValueError):
            layer_with_bias.load_weights(str(tmp_dir / "layer_without_bias.json"))
        

def test_MLP():
    pass


def test_MLP_zero_grad():
    pass


def test_MLP_save_and_load():
    pass