from tinygpt.buffer import Buffer
from tinygpt.mlops import Sum


def test_Sum():
    a = Buffer([1., 2., 3., 4.])
    b = Buffer([5., 6., 7., 8.])

    for incoming_grad in [Buffer([-1., -2., -3.]), Buffer([3.14])]:
        options = [[False, False], [True, False], [False, True], [True, True]]
        for needs_input_grad in options:
            # Only a needs input grad
            sum_op = Sum(needs_input_grad=needs_input_grad)

            # Do the forward pass
            output_buffer = sum_op.forward(a, b)

            assert output_buffer == a + b

            # Do the backward pass
            backward_gradient_buffers = sum_op.backward(incoming_grad)

            for i in range(2):
                if needs_input_grad[i]:
                    assert backward_gradient_buffers[i] == incoming_grad
                else:
                    assert backward_gradient_buffers[i] is None
