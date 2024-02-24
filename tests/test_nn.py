import pytest

from tinygpt.tensor import Tensor
from tinygpt.module import Module
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

            # Check weights of the layer have gradient
            for p in layer_without_bias.parameters().values():
                assert p.grad is not None
                assert p.grad.shape == p.shape

            for p in layer_with_bias.parameters().values():
                assert p.grad is not None
                assert p.grad.shape == p.shape

            # Call zero_grad()
            layer_without_bias.zero_grad()
            layer_with_bias.zero_grad()

            # Check weights of the layer have gradient
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

        # Check trainable parameters returns no trainable parameters
        assert len(layer_without_bias.trainable_parameters()) == 0
        assert len(layer_with_bias.trainable_parameters()) == 0

        # Unfreeze the weights
        layer_without_bias.unfreeze()
        layer_with_bias.unfreeze()

        # Check trainable parameters returns all weights
        assert len(layer_without_bias.trainable_parameters()) == 1
        assert len(layer_with_bias.trainable_parameters()) == 2


def test_FullyConnectedLayer_train_and_eval():
    for input_dims, output_dims in [(12, 12), (12, 6), (12, 24)]:
        # Create the layers with different configurations
        layer_without_bias = FullyConnectedLayer(input_dims=input_dims, output_dims=output_dims, bias=False)
        layer_with_bias = FullyConnectedLayer(input_dims=input_dims, output_dims=output_dims, bias=True)

        # Set the layers in eval mode
        layer_without_bias.eval()
        layer_with_bias.eval()

        # Check layers are not in training mode
        assert not layer_without_bias.training
        assert not layer_with_bias.training

        # Set the layers in training model
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
        input_tensor = Tensor.uniform((12, 3, input_dims), requires_grad=True)

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
    # Wrong data types
    with pytest.raises(TypeError):
        _ = MLP(input_dims=None, hidden_dims=[12], activation_fn='relu', bias=True)
    with pytest.raises(TypeError):
        _ = MLP(input_dims=12, hidden_dims=None, activation_fn='relu', bias=True)
    with pytest.raises(ValueError):
        _ = MLP(input_dims=12, hidden_dims=[12], activation_fn='unknown_fn', bias=True)

    # Wrong values
    for wrong_value in [-1, 0]:
        with pytest.raises(ValueError):
            _ = MLP(input_dims=wrong_value, hidden_dims=[12], activation_fn='relu', bias=True)
        with pytest.raises(ValueError):
            _ = MLP(input_dims=12, hidden_dims=[wrong_value], activation_fn='relu', bias=True)
        with pytest.raises(ValueError):
            _ = MLP(input_dims=12, hidden_dims=[12], activation_fn=wrong_value, bias=True)

    # Do the inference
    for activation_fn in MLP.activation_functions.keys():
        for input_dims, hidden_dims in [(3, [3, 3, 3]), (3, [6, 2, 2]), (3, [5, 4, 5])]:
            
            mlp_without_bias = MLP(
                input_dims=input_dims, hidden_dims=hidden_dims, activation_fn=activation_fn, bias=False
            )
            mlp_with_bias = MLP(
                input_dims=input_dims, hidden_dims=hidden_dims, activation_fn=activation_fn, bias=True
            )

            # Check the modules have the expected components
            assert len(mlp_without_bias.children()) == 1
            assert len(mlp_with_bias.children()) == 1

            assert len(mlp_without_bias.modules()) == len(hidden_dims) + 1
            assert len(mlp_with_bias.modules()) == len(hidden_dims) + 1

            assert len(mlp_without_bias.named_modules()) == len(hidden_dims) + 1
            assert len(mlp_with_bias.named_modules()) == len(hidden_dims) + 1

            assert len(mlp_without_bias.parameters()) == 1
            assert len(mlp_with_bias.parameters()) == 1
            
            assert len(mlp_without_bias.trainable_parameters()) == 1
            assert len(mlp_with_bias.trainable_parameters()) == 1
            
            assert len(mlp_without_bias.leaf_modules()) == 1
            assert len(mlp_with_bias.leaf_modules()) == 1
            
            # Check the parameters have the expected name and shape
            for key_layers, layers in mlp_with_bias.parameters().items():
                assert key_layers == 'layers'
                for idx_layer, p_dict in enumerate(layers):
                    in_d = input_dims if idx_layer == 0 else hidden_dims[idx_layer - 1]
                    for name_p, p in p_dict.items(): 
                        assert name_p in ('weights', 'bias')

                        if name_p == 'weights':
                            assert p.shape == (in_d, hidden_dims[idx_layer])
                        else:
                            assert p.shape == (hidden_dims[idx_layer],)
                        
                        assert p.requires_grad

            # Use different shapes for the input tensor
            for input_shape in [(input_dims,), (24, input_dims), (12, 3, input_dims)]:
                input_tensor = Tensor.uniform(input_shape, requires_grad=True)

                # Do inference
                output_tensor_without_bias = mlp_without_bias(input_tensor)
                output_tensor_with_bias = mlp_with_bias(input_tensor)

                # Check the output
                assert output_tensor_without_bias.shape == (*input_shape[:-1], hidden_dims[-1])
                assert output_tensor_with_bias.shape == (*input_shape[:-1], hidden_dims[-1])

                assert output_tensor_without_bias.requires_grad
                assert output_tensor_with_bias.requires_grad

                # Do inference with a wrong shape for the input
                with pytest.raises(RuntimeError):
                    _ = mlp_without_bias(Tensor.uniform((*input_shape[:-1], input_shape[-1] + 1)))

                with pytest.raises(RuntimeError):
                    _ = mlp_with_bias((Tensor.uniform((*input_shape[:-1], input_shape[-1] + 1))))


def test_MLP_zero_grad():
    # Call zero_grad after backward pass
    for activation_fn in MLP.activation_functions.keys():
        for input_dims, hidden_dims in [(3, [3, 3, 3]), (3, [6, 2, 2]), (3, [5, 4, 5])]:
            # Create the MLP with different configurations
            mlp_without_bias = MLP(
                input_dims=input_dims, hidden_dims=hidden_dims, activation_fn=activation_fn, bias=False
            )
            mlp_with_bias = MLP(
                input_dims=input_dims, hidden_dims=hidden_dims, activation_fn=activation_fn, bias=True
            )

            # Do the forward and backward pass
            for input_shape in [(input_dims,), (24, input_dims), (12, 3, input_dims)]:
                input_tensor = Tensor.uniform(input_shape, requires_grad=True)

                # Do the forward pass
                output_tensor_without_bias = mlp_without_bias(input_tensor)
                output_tensor_with_bias = mlp_with_bias(input_tensor)

                # Do the backward pass
                output_tensor_without_bias.sum(axes=tuple(i for i in range(output_tensor_without_bias.ndim))).backward()
                output_tensor_with_bias.sum(axes=tuple(i for i in range(output_tensor_with_bias.ndim))).backward()

                # Check weights of the MLP have gradient
                for layers in mlp_without_bias.parameters().values():
                    for layer in layers:
                        for p in layer.values(): 
                            assert p.grad is not None
                            assert p.grad.shape == p.shape

                for layers in mlp_with_bias.parameters().values():
                    for layer in layers:
                        for p in layer.values(): 
                            assert p.grad is not None
                            assert p.grad.shape == p.shape

                # Call zero_grad()
                mlp_without_bias.zero_grad()
                mlp_with_bias.zero_grad()

                # Check weights of the MLP have no gradient
                for layers in mlp_without_bias.parameters().values():
                    for layer in layers:
                        for p in layer.values(): 
                            assert p.grad is None

                for layers in mlp_with_bias.parameters().values():
                    for layer in layers:
                        for p in layer.values(): 
                            assert p.grad is None


def test_MLP_freeze_and_unfreeze():
    for activation_fn in MLP.activation_functions.keys():
        for input_dims, hidden_dims in [(12, [12, 12, 12]), (12, [6, 2, 2]), (12, [24, 12, 24])]:
            
            # Create the MLP with different configurations
            mlp_without_bias = MLP(
                input_dims=input_dims, hidden_dims=hidden_dims, activation_fn=activation_fn, bias=False
            )
            mlp_with_bias = MLP(
                input_dims=input_dims, hidden_dims=hidden_dims, activation_fn=activation_fn, bias=True
            )

            # Freeze the weights
            mlp_without_bias.freeze()
            mlp_with_bias.freeze()

            # Check trainable parameters returns no trainable parameters
            assert mlp_without_bias.trainable_parameters() == {'layers': [{}] * len(hidden_dims)}
            assert mlp_with_bias.trainable_parameters() == {'layers': [{}] * len(hidden_dims)}

            # Unfreeze the weights
            mlp_without_bias.unfreeze()
            mlp_with_bias.unfreeze()

            # Check trainable parameters returns all the weights
            for layers in mlp_without_bias.trainable_parameters().values():
                assert len(layers) == len(hidden_dims)
                for layer in layers:
                    assert len(layer) == 1

            for layers in mlp_with_bias.trainable_parameters().values():
                assert len(layers) == len(hidden_dims)
                for layer in layers:
                    assert len(layer) == 2


def test_MLP_train_and_eval():
    for activation_fn in MLP.activation_functions.keys():
        for input_dims, hidden_dims in [(12, [12, 12, 12]), (12, [6, 2, 2]), (12, [24, 12, 24])]:
            # Create the MLP with different configurations
            mlp_without_bias = MLP(
                input_dims=input_dims, hidden_dims=hidden_dims, activation_fn=activation_fn, bias=False
            )
            mlp_with_bias = MLP(
                input_dims=input_dims, hidden_dims=hidden_dims, activation_fn=activation_fn, bias=True
            )

            # Set the MLPs in eval mode
            mlp_without_bias.eval()
            mlp_with_bias.eval()

            # Check layers are not in training mode
            assert not mlp_without_bias.training
            assert not mlp_with_bias.training

            # Set the MLPs in training mode
            mlp_without_bias.train()
            mlp_with_bias.train()

            # Check MLPs are in training mode
            assert mlp_without_bias.training
            assert mlp_with_bias.training


def test_MLP_save_and_load(tmp_path):
    # Create different MLPs and save and restore it
    tmp_dir = tmp_path / "test_MLP_save_and_load"
    tmp_dir.mkdir()
    for activation_fn in MLP.activation_functions.keys():
        for input_dims, hidden_dims in [(3, [3, 3, 3]), (3, [6, 2, 2]), (3, [5, 4, 5])]:
            # Create the MLP with different configurations
            mlp_without_bias = MLP(
                input_dims=input_dims, hidden_dims=hidden_dims, activation_fn=activation_fn, bias=False
            )
            mlp_with_bias = MLP(
                input_dims=input_dims, hidden_dims=hidden_dims, activation_fn=activation_fn, bias=True
            )

            # Do the forward and backward pass
            input_tensor = Tensor.uniform((12, 3, input_dims), requires_grad=True)

            # Do the forward pass
            output_tensor_without_bias = mlp_without_bias(input_tensor)
            output_tensor_with_bias = mlp_with_bias(input_tensor)

            # Save the layers
            mlp_without_bias.save_weights(str(tmp_dir / "mlp_without_bias.json"))
            mlp_with_bias.save_weights(str(tmp_dir / "mlp_with_bias.json"))

            # Create the MLPs again
            mlp_without_bias = MLP(
                input_dims=input_dims, hidden_dims=hidden_dims, activation_fn=activation_fn, bias=False
            )
            mlp_with_bias = MLP(
                input_dims=input_dims, hidden_dims=hidden_dims, activation_fn=activation_fn, bias=True
            )

            # Load the weights
            mlp_without_bias.load_weights(str(tmp_dir / "mlp_without_bias.json"))
            mlp_with_bias.load_weights(str(tmp_dir / "mlp_with_bias.json"))

            # Do the inference again
            new_output_tensor_without_bias = mlp_without_bias(input_tensor)
            new_output_tensor_with_bias = mlp_with_bias(input_tensor)

            # Check results are equal to the results before
            assert all(new_output_tensor_without_bias.buffer == output_tensor_without_bias.buffer)
            assert all(new_output_tensor_with_bias.buffer == output_tensor_with_bias.buffer)

            # Try to load wrong weight file
            with pytest.raises(ValueError):    
                mlp_without_bias.load_weights(str(tmp_dir / "mlp_with_bias.json"))
            
            with pytest.raises(ValueError):
                mlp_with_bias.load_weights(str(tmp_dir / "mlp_without_bias.json"))


def test_freeze_and_unfreeze():
    # Create a fake class to test the methods
    class Temp(Module):
        def __init__(self):
            super().__init__()
            # Private members
            self._private_tensor = Tensor.uniform((1, 3), requires_grad=False)
            self._private_list = [
                Tensor.uniform((1, 4), requires_grad=False), Tensor.uniform((1, 5), requires_grad=False)
            ]
            self._private_dict = dict(
                first=Tensor.uniform((1, 6), requires_grad=False), second=Tensor.uniform((1, 7), requires_grad=False)
            )

            # Normal trainable members
            self.trainable_tensor = Tensor.uniform((1, 8), requires_grad=True)
            self.trainable_list = [
                Tensor.uniform((1, 9), requires_grad=True), Tensor.uniform((1, 10), requires_grad=True)
            ]
            self.trainable_dict = dict(
                first=Tensor.uniform((1, 11), requires_grad=True), second=Tensor.uniform((1, 12), requires_grad=True)
            )

            # Normal non-trainable members
            self.non_trainable_tensor = Tensor.uniform((1, 13), requires_grad=False)
            self.non_trainable_list = [
                Tensor.uniform((1, 14), requires_grad=False), Tensor.uniform((1, 15), requires_grad=False)
            ]
            self.non_trainable_dict = dict(
                first=Tensor.uniform((1, 16), requires_grad=False), second=Tensor.uniform((1, 17), requires_grad=False)
            )

        def __call__(self, x: Tensor) -> Tensor:
            output = x
            output = (self.trainable_tensor + output).sum((0,1))

            for tensor in self.trainable_list:
                output = (tensor + output).sum((0,1))
            
            for tensor in self.trainable_dict.values():
                output = (tensor + output).sum((0,1))

            return output
    

    # Create an object and validate all the parameters
    temp = Temp()

    # Check all parameters
    assert len(temp.parameters()) == 6

    # Check trainable parameters
    trainable_tensors = 0
    for k, v in temp.trainable_parameters().items():
        assert not k.startswith("_")
        if isinstance(v, Tensor):
            assert not k.startswith("non_trainable")
            trainable_tensors += 1
        elif isinstance(v, dict):
            for tensor in v.values():
                if isinstance(tensor, Tensor):
                    assert not k.startswith("non_trainable") 
                    trainable_tensors += 1
        elif isinstance(v, list):
            for tensor in v:
                if isinstance(tensor, Tensor):
                    assert not k.startswith("non_trainable") 
                    trainable_tensors += 1

    assert trainable_tensors == 5

    # Freeze all of them
    temp.freeze()

    assert len(temp.parameters()) == 6 
    for k, v in temp.trainable_parameters().items():
        assert not k.startswith("_")
        assert not isinstance(v, Tensor)
        if isinstance(v, dict):
            for tensor in v.values():
                assert not isinstance(v, Tensor)
        elif isinstance(v, list):
            for tensor in v:
                assert not isinstance(v, Tensor)

    # Unfreeze all of them
    temp.unfreeze()

    assert len(temp.parameters()) == 6 
    trainable_tensors = 0
    for k, v in temp.trainable_parameters().items():
        assert not k.startswith("_")
        if isinstance(v, Tensor):
            trainable_tensors += 1
        elif isinstance(v, dict):
            for tensor in v.values():
                if isinstance(tensor, Tensor):
                    trainable_tensors += 1
        elif isinstance(v, list):
            for tensor in v:
                if isinstance(tensor, Tensor):
                    trainable_tensors += 1

    assert trainable_tensors == 10

    # Unfreeze those that starts with 'non_trainable'
    temp.freeze()
    temp.unfreeze(
        keys=[
            'trainable_tensor', 'trainable_list', 'trainable_dict', 
            'trainable_list.0', 'trainable_list.1', 
            'trainable_dict.first', 'trainable_dict.second'
        ]
    )

    assert len(temp.parameters()) == 6
    trainable_tensors = 0
    for k, v in temp.trainable_parameters().items():
        assert not k.startswith("_")
        if isinstance(v, Tensor):
            assert not k.startswith("non_trainable")
            trainable_tensors += 1
        elif isinstance(v, dict):
            for tensor in v.keys():
                if isinstance(tensor, Tensor):
                    assert not k.startswith("non_trainable") 
                    trainable_tensors += 1
        elif isinstance(v, list):
            for tensor in v:
                if isinstance(tensor, Tensor):
                    assert not k.startswith("non_trainable") 
                    trainable_tensors += 1

    assert trainable_tensors == 3

    # Freeze some parameters
    temp.freeze(keys='trainable_tensor')

    assert len(temp.parameters()) == 6
    trainable_tensors = 0
    for k, v in temp.trainable_parameters().items():
        assert not k.startswith("_")
        assert not isinstance(v, Tensor)
        if isinstance(v, dict):
            for tensor in v.values():
                if isinstance(tensor, Tensor):
                    assert not k.startswith("non_trainable") 
                    trainable_tensors += 1
        elif isinstance(v, list):
            for tensor in v:
                if isinstance(tensor, Tensor):
                    assert not k.startswith("non_trainable") 
                    trainable_tensors += 1

    assert trainable_tensors == 4

    # Do a forward pass
    result = temp(Tensor.uniform((1, 1)))

    # Check the gradient is empty
    assert temp.trainable_tensor.grad is None
    assert all([tensor.grad is None for tensor in temp.trainable_list])
    assert all([tensor.grad is None for tensor in temp.trainable_dict.values()])
    assert temp.non_trainable_tensor.grad is None
    assert all([tensor.grad is None for tensor in temp.non_trainable_list])
    assert all([tensor.grad is None for tensor in temp.non_trainable_dict.values()])

    # Do the backward pass 
    result.backward()

    # Check all parameters have the expected gradient
    assert temp.trainable_tensor.grad is None
    assert all([tensor.grad is not None for tensor in temp.trainable_list])
    assert all([tensor.grad is not None for tensor in temp.trainable_dict.values()])
    assert temp.non_trainable_tensor.grad is None
    assert all([tensor.grad is None for tensor in temp.non_trainable_list])
    assert all([tensor.grad is None for tensor in temp.non_trainable_dict.values()])