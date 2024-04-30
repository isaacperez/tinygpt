import copy

import pytest

from tinygpt.nn import MLP
from tinygpt.tensor import Tensor
from tinygpt.module import Module
from tinygpt.optimizers import SGD
from tinygpt.utils import tree_flatten


def test_SGD_with_MLP():
    # Create a MLP to test the optimizer
    mlp = MLP(input_dims=4, hidden_dims=[4, 4], activation_fn='relu', bias=True)

    # Create the optimizer
    sgd_optimizer = SGD(module=mlp, learning_rate=0.1, momentum=0.9, weight_decay=0.1, dampening=0.1, nesterov=False)

    # Do one update without gradients
    trainable_parameters_before_update = copy.deepcopy(mlp.trainable_parameters())
    sgd_optimizer.update()
    assert trainable_parameters_before_update == mlp.trainable_parameters()

    # Do a forward and backward pass
    sgd_optimizer.zero_grad()
    output = mlp(Tensor.zeros((2, 4)))
    output.sum((0, 1)).backward()

    # Apply the gradients
    sgd_optimizer.update()

    # Check the weights have changed
    old_weights = tree_flatten(trainable_parameters_before_update)
    new_weights = tree_flatten(mlp.trainable_parameters())
    for (old_n, old_tensor), (new_n, new_tensor) in zip(old_weights, new_weights):
        assert old_n == new_n 
        assert old_tensor.shape == new_tensor.shape

        # After the update, the tensors should be different
        assert new_tensor.to_python() != old_tensor.to_python()


def test_SGD():
    # Create a dummy module with a single tensor to track the value update during some iterations
    class ToyModule(Module):
        def __init__(self):
            super().__init__()
            self.tensor = Tensor([1.2, 2.0], requires_grad=True)

        def __call__(self, x: Tensor) -> Tensor:
            return x.dot(self.tensor)


    module = ToyModule()

    # Create the optimizer
    sgd_optimizer = SGD(module=module, learning_rate=0.1, momentum=0.9, weight_decay=0.0, dampening=0.0, nesterov=False)

    # Train the model
    current_version = module.tensor._version
    expected_outputs = [Tensor(3.2), Tensor(3.0), Tensor(2.62), Tensor(2.078)]
    expected_new_tensors = [Tensor([1.1, 1.9]), Tensor([0.91, 1.71]), Tensor([0.639, 1.439]), Tensor([0.2951, 1.0951])]
    expected_grad = Tensor.ones((2,))
    for it in range(4):
        sgd_optimizer.zero_grad()

        output = module(Tensor.ones((2,)))
        output.backward()

        sgd_optimizer.update()

        # After each update, the version of the tensor increases
        assert module.tensor._version > current_version
        current_version = module.tensor._version

        # Check the values
        delta = output.buffer - expected_outputs[it].buffer
        assert all(delta > -1e-05)
        assert all(delta < 1e-05)

        delta = module.tensor.buffer - expected_new_tensors[it].buffer
        assert all(delta > -1e-05)
        assert all(delta < 1e-05)

        delta = module.tensor.grad.buffer - expected_grad.buffer
        assert all(delta > -1e-05)
        assert all(delta < 1e-05)


def test_save_and_load_optimizer(tmp_path):
    # Create a temp folder
    tmp_dir = tmp_path / "test_save_and_load_optimizer"
    tmp_dir.mkdir()

    # Create a MLP to test the optimizer
    mlp = MLP(input_dims=4, hidden_dims=[4, 4], activation_fn='relu', bias=True)

    # Create the optimizer
    sgd = SGD(module=mlp, learning_rate=0.1, momentum=0.9, weight_decay=0.1, dampening=0.1, nesterov=False)

    # Do a forward and backward pass
    sgd.zero_grad()
    output = mlp(Tensor.zeros((2, 4)))
    output.sum((0, 1)).backward()

    # Apply the gradients
    sgd.update() 

    # Store the state
    sgd.save_state(str(tmp_dir / "sgd.json"))

    # Create a new optimizer
    new_sgd = SGD(module=mlp, learning_rate=0.1, momentum=0.9, weight_decay=0.1, dampening=0.1, nesterov=False)

    assert tree_flatten(new_sgd.state) != tree_flatten(sgd.state)

    # Load the state into the new optimizer with strict mode (it should fail because the state is empty)
    with pytest.raises(ValueError): 
        new_sgd.load_state(str(tmp_dir / "sgd.json"))

    # Load the state without strict mode
    new_sgd.load_state(str(tmp_dir / "sgd.json"), strict=False)

    assert tree_flatten(new_sgd.state) == tree_flatten(sgd.state)
