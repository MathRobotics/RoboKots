import numpy as np

from robokots.core.axis_tensor import AlgorithmSpec, AxisTensor, LayoutPolicy


def test_axis_tensor_to_axes_returns_transposed_view():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    tensor = AxisTensor(data, ("time", "joint", "coord"))

    actual = tensor.to_axes("joint", "time", "coord")

    assert actual.axes == ("joint", "time", "coord")
    assert actual.shape == (3, 2, 4)
    assert np.shares_memory(actual.data, data)
    np.testing.assert_array_equal(actual.data, np.transpose(data, (1, 0, 2)))


def test_axis_tensor_to_layout_materializes_contiguous_data():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    tensor = AxisTensor(data, ("time", "joint", "coord"))

    actual = tensor.to_layout(
        LayoutPolicy(
            axes=("joint", "time", "coord"),
            contiguous=True,
            backend="numpy",
            memory_order="C",
        )
    )

    assert actual.axes == ("joint", "time", "coord")
    assert actual.layout.backend == "numpy"
    assert actual.layout.contiguous
    assert actual.data.flags.c_contiguous
    assert not np.shares_memory(actual.data, data)


def test_axis_tensor_prepare_for_algorithm_spec():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    tensor = AxisTensor(data, ("time", "joint", "coord"))
    spec = AlgorithmSpec(
        required_axes=("time", "coord", "joint"),
        layout_policy=LayoutPolicy(
            axes=("time", "coord", "joint"),
            contiguous=True,
            backend="numpy",
        ),
    )

    actual = tensor.prepare_for(spec)

    assert actual.axes == ("time", "coord", "joint")
    assert actual.layout.contiguous
    np.testing.assert_array_equal(actual.data, np.transpose(data, (0, 2, 1)))
