"""Public RoboKots cache reaches the compiled ABA across dependency changes."""
from pathlib import Path

import numpy as np

from robokots.kots import Kots


def test_compiled_inward_cache_updates_position_velocity_and_gravity(monkeypatch):
    path = Path(__file__).parent / "test_model" / "branched_fixed.urdf"
    kots = Kots.from_urdf_file(str(path), order=3, backend="rust")
    q = np.linspace(-0.2, 0.3, kots.dof())
    v = np.linspace(0.1, -0.2, kots.dof())
    acceleration = np.linspace(0.3, -0.1, kots.dof())
    gravity = np.array([0.2, -0.3, -9.81])
    states = [
        (q, v, gravity),
        (q, v + 0.2, gravity),
        (q, v + 0.2, np.zeros(3)),
        (q + 0.1, v, gravity),
        (q, v, gravity),
    ]
    torques = [kots.inverse_dynamics(x, y, acceleration, gravity=g, backend="numpy")
               for x, y, g in states]

    def forbidden(*args, **kwargs):
        raise AssertionError("Rust inward cache must not fall back to Python dynamics")

    monkeypatch.setattr(kots, "inverse_dynamics", forbidden)
    cache = kots.create_inward_cache(backend="rust")
    for (position, velocity, g), torque in zip(states, torques):
        cache.prepare(position, velocity, g)
        np.testing.assert_allclose(cache.forward_dynamics(torque), acceleration,
                                   atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(
            kots.forward_dynamics(position, velocity, torque, gravity=g, backend="rust"),
            acceleration, atol=2e-10, rtol=2e-10,
        )
