from tinygpt.tensor import Tensor


def test_Tensor():

    # Create int scalars
    _ = Tensor(-1, requires_grad=False)
    _ = Tensor(1, requires_grad=False)

    # Create float scalars
    _ = Tensor(-2.0, requires_grad=False)
    _ = Tensor(2.0, requires_grad=False)

    # Create tensor with lists
    _ = Tensor([-3, 4], requires_grad=False)
    _ = Tensor([-3.0, 4.0], requires_grad=False)
