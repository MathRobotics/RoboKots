"""Measure public world-motion reads and cached mixed derivatives before/after migration."""
import argparse
import json
from pathlib import Path
import platform
import hashlib
import numpy as np
from robokots.kots import Kots, StateType
from .core_state_layout import measure


def run(output, reference=False):
    if reference:
        from robokots.outward.rust.data import RustOutwardState
        from robokots.core.kernels.cmtm_apply import world_spatial_value
        original = RustOutwardState.state_value
        def read_reference(self, state):
            if state.frame_name == "world" and state.data_type in ("vel", "acc", "jerk", "snap", "crackle"):
                n = state.key_order - 1
                link = state.owner_name if state.owner_type == "link" else self.robot.links[self.robot.joint(state.owner_name).child_link_id].name
                return world_spatial_value(self.cmtm(state.owner_type, state.owner_name, n+1), self.cmtm("link", link, n), n)
            return original(self, state)
        RustOutwardState.state_value = read_reference
    root = Path(__file__).resolve().parents[2]
    model = root / "tests/test_model/branched_fixed.urdf"
    results, values = {}, {}
    for order in (3, 6):
        for shape in ((), (2, 3)):
            k = Kots.from_urdf_file(str(model), order=order, backend="rust")
            rng = np.random.default_rng(944)
            motion = rng.normal(scale=.2, size=shape + (k.dof()*order,))
            k.import_motions(motion)
            k.dynamics(gravity=[.2, -.3, -9.81])
            keys = ["vel", "acc", "jerk", "snap", "crackle"][:order-1]
            states = [StateType(owner, name, key, "world") for owner, name in
                      [("link", "a_tip"), ("joint", "a_elbow"), ("joint", "b_payload_fixed")]
                      for key in keys if not (owner == "joint" and key == "jerk")]
            jac = k.jacobian(states)
            v = rng.normal(size=shape + (jac.shape[-1],))
            w = rng.normal(size=shape + (jac.shape[-2],))
            ops = {"world_values": lambda: k.state_info_list(states),
                   "dense_cached": lambda: k.jacobian(states),
                   "jvp_cached": lambda: k.jacobian_mul(states, v),
                   "vjp_cached": lambda: k.jacobian_transpose_mul(states, w)}
            for name, fn in ops.items():
                label = f"order{order}/batch{shape}/{name}"
                results[label] = measure(fn, 10, 20, 10)
                values[label] = np.asarray(fn()).tolist()
    import robokots._rust_core as native
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
                        "extension_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()},
        "workload": {"model": str(model.relative_to(root)), "seed": 944, "orders": [3, 6], "batch_shapes": [[], [2, 3]],
                     "gravity": [.2, -.3, -9.81], "dtype": "float64", "warmup": 10, "samples": 20, "calls": 10,
                     "notes": "State already computed; public Python/PyO3/NumPy costs included; cached derivatives; no JIT"},
        "reference_python_world_transform": reference, "timings": results, "outputs": values}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true", help="Use the previous Python world-transform implementation")
    args = parser.parse_args()
    run(args.output, args.reference)
