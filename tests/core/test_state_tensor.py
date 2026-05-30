import numpy as np

from robokots.core.state_tensor import JacobianTensor, StateTensor


def test_state_tensor_from_array_assigns_batch_and_state_axes():
    data = np.zeros((2, 3, 12))

    tensor = StateTensor.from_array(data, state_types=("vel", "acc"))

    assert tensor.axes == ("batch0", "batch1", "state")
    assert tensor.shape == (2, 3, 12)
    assert tensor.batch_shape == (2, 3)
    assert tensor.state_dim == 12
    assert tensor.state_types == ("vel", "acc")


def test_state_tensor_from_unbatched_array_has_only_state_axis():
    data = np.zeros(12)

    tensor = StateTensor.from_array(data)

    assert tensor.axes == ("state",)
    assert tensor.batch_shape == ()
    assert tensor.state_dim == 12


def test_jacobian_tensor_from_array_assigns_state_and_motion_axes():
    data = np.zeros((2, 3, 12, 18))

    tensor = JacobianTensor.from_array(data, state_types=("vel", "acc"))

    assert tensor.axes == ("batch0", "batch1", "state", "motion")
    assert tensor.shape == (2, 3, 12, 18)
    assert tensor.batch_shape == (2, 3)
    assert tensor.state_dim == 12
    assert tensor.motion_dim == 18
    assert tensor.state_types == ("vel", "acc")


def test_jacobian_tensor_from_unbatched_array_has_state_motion_axes():
    data = np.zeros((12, 18))

    tensor = JacobianTensor.from_array(data)

    assert tensor.axes == ("state", "motion")
    assert tensor.batch_shape == ()
    assert tensor.state_dim == 12
    assert tensor.motion_dim == 18
