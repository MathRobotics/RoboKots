"""Cold-start NumPy operation with native imports unavailable."""
import os
from pathlib import Path
import subprocess
import sys


def test_numpy_without_rust_in_fresh_process():
    root = Path(__file__).resolve().parents[1]
    script = r'''
import importlib.abc
import sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "robokots._rust" or fullname.startswith("robokots._rust."):
            raise AssertionError("NumPy attempted to import Rust: " + fullname)
sys.meta_path.insert(0, NoNative())
import numpy as np
from robokots.kots import Kots, StateType
for order in (3, 7):
    k = Kots.from_urdf_file(sys.argv[1], order=order, backend="numpy")
    rng = np.random.default_rng(23)
    motion = rng.normal(scale=.2, size=k.dof()*order)
    gravity = [.2, -.3, -9.81]
    k.import_motions(motion)
    k.dynamics(gravity=gravity)
    states = [StateType("total_joint", "total_joint", "torque" if order == 3 else "torque_diff4"),
              StateType("link", "a_tip", "momentum", "world"),
              StateType("link", "a_tip", "force", "world")]
    assert np.isfinite(k.state_info_list(states)).all()
    j = k.jacobian(states)
    v = rng.normal(size=j.shape[-1]); w = rng.normal(size=j.shape[-2])
    np.testing.assert_allclose(k.jacobian_mul(states, v), j @ v, atol=1e-9)
    np.testing.assert_allclose(k.jacobian_transpose_mul(states, w), j.T @ w, atol=1e-9)
    q, vel, acc = motion.reshape(k.dof(), order)[:, :3].T
    torque = k.inverse_dynamics(q, vel, acc, gravity=gravity)
    np.testing.assert_allclose(k.forward_dynamics(q, vel, torque, gravity=gravity), acc, atol=1e-9)
assert not any(name.startswith("robokots._rust") for name in sys.modules)
'''
    env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = subprocess.run([sys.executable, "-c", script, str(root / "tests/test_model/branched_fixed.urdf")],
                            cwd=root, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
