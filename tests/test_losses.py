import pytest

from tinygpt.losses import CrossEntropyLoss
from tinygpt.tensor import Tensor


def test_CrossEntropyLoss():

    loss_fn = CrossEntropyLoss()

    # 1D loss
    logits = Tensor([1.0, 5.0, 1.0])
    labels = Tensor([0.0, 1.0, 0.0])
    loss = loss_fn(logits=logits, labels=labels)

    assert loss.shape == ()
    delta = loss.buffer - Tensor(0.035976).buffer
    assert all(delta > -1e-05)
    assert all(delta < 1e-05)

    # 2D loss
    logits = Tensor([[1.0, 5.0, 1.0], [-1.0, 5.0, -1.0]])
    labels = Tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    loss = loss_fn(logits=logits, labels=labels)

    assert loss.shape == (2,)
    delta = loss.buffer - Tensor([0.035976, 6.004945]).buffer
    assert all(delta > -1e-05)
    assert all(delta < 1e-05)
    
    # Broadcasting
    logits = Tensor([[1.0, 5.0, 1.0], [-1.0, 5.0, -1.0]])
    labels = Tensor([0.0, 1.0, 0.0])
    loss = loss_fn(logits=logits, labels=labels)

    assert loss.shape == (2,)
    delta = loss.buffer - Tensor([0.035976, 0.004945]).buffer
    assert all(delta > -1e-05)
    assert all(delta < 1e-05)

    # 3D loss
    logits = Tensor([[[1.0, 5.0, 1.0], [-1.0, 5.0, -1.0]]])
    labels = Tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])

    with pytest.raises(RuntimeError):
        loss = loss_fn(logits=logits, labels=labels)

    # 0D loss
    logits = Tensor(3.4)
    labels = Tensor(1.0)

    with pytest.raises(RuntimeError):
        loss = loss_fn(logits=logits, labels=labels)