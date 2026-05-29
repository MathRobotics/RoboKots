import numpy as np
import pytest


matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robokots.core.viz import (
    set_equall_aspect,
    set_equall_aspect_3d,
    show_link_points,
)


def test_set_equall_aspect_uses_margin_argument():
    fig, ax = plt.subplots()
    data = np.array([[0.0, 0.0], [1.0, 3.0]])

    set_equall_aspect(ax, data, margin=0.5)

    np.testing.assert_allclose(ax.get_xlim(), (-1.5, 2.5))
    np.testing.assert_allclose(ax.get_ylim(), (-0.5, 3.5))
    plt.close(fig)


def test_set_equall_aspect_3d_uses_margin_argument():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    data = np.array([[0.0, 0.0, 0.0], [1.0, 3.0, 5.0]])

    set_equall_aspect_3d(ax, data, margin=0.5)

    np.testing.assert_allclose(ax.get_xlim3d(), (-2.5, 3.5))
    np.testing.assert_allclose(ax.get_ylim3d(), (-1.5, 4.5))
    np.testing.assert_allclose(ax.get_zlim3d(), (-0.5, 5.5))
    plt.close(fig)


def test_show_link_points_plots_z_coordinates_on_3d_axes(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    link_pos = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

    show_link_points(link_pos, ax=ax, dimension=3)

    _, _, z_data = ax.collections[-1]._offsets3d
    np.testing.assert_allclose(np.asarray(z_data), link_pos[:, 2])
    plt.close(fig)
