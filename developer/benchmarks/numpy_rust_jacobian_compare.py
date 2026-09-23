"""Same-build NumPy/Rust dense Jacobian and direct-product comparison."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import platform
from unittest.mock import patch

import numpy as np

from robokots import outward as outward_api
from robokots.kots import Kots, StateType
from .core_state_layout import measure

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "tests/test_model/branched_fixed.urdf"


def serial_model(dof):
    """Synthetic serial revolute chain with a fixed tool, not a commercial robot."""
    links = [{"id": 0, "name": "world"}]
    joints = []
    axes = ([0., 0., 1.], [0., 1., 0.], [1., 0., 0.])
    for i in range(dof):
        links.append({"id": i+1, "name": f"link{i+1}", "mass": 1.,
                      "cog": [0., 0., .08], "inertia": {"ixx": .01, "iyy": .01, "izz": .01, "ixy": 0., "ixz": 0., "iyz": 0.}})
        angle = .1*(-1)**i
        joints.append({"id": i, "name": f"joint{i+1}", "type": "revolute", "axis": axes[i%3],
                       "parent_link_id": i, "child_link_id": i+1,
                       "origin": {"position": [.03*(-1)**i, .02, .18+.015*i],
                                  "orientation": [float(np.cos(angle/2)), 0., float(np.sin(angle/2)), 0.]}})
    links.append({"id": dof+1, "name": "a_tip", "mass": .2})
    joints.append({"id": dof, "name": "tool_fixed", "type": "fixed",
                   "parent_link_id": dof, "child_link_id": dof+1,
                   "origin": {"position": [.1, 0., .06]}})
    return {"schema_version": "0.0.2", "links": links, "joints": joints}


def main(output=None, case=None, serial_dof=None):
    if serial_dof is not None and (serial_dof < 1 or case not in ("position", "pose", "kinematics", "torque")):
        raise ValueError("--serial-dof requires a positive DOF and --case position/pose/kinematics/torque")
    model_data = serial_model(serial_dof) if serial_dof is not None else None
    model_name = f"synthetic serial {serial_dof}-DOF revolute chain with fixed tool" if model_data else str(MODEL.relative_to(ROOT))
    rows = []
    seed, warmup, samples, calls = 20260923, 5, 15, 3
    gravity = [.2, -.3, -9.81]
    cases = [("position", 1, ()), ("pose", 1, ()), ("kinematics", 3, ()), ("torque", 3, ()),
             ("mixed", 3, ()), ("mixed", 3, (6,)), ("mixed", 6, ())]
    if case is not None:
        cases = [entry for entry in cases if entry[0] == case]
    for family, order, shape in cases:
        rng = np.random.default_rng(seed)
        if family == "position":
            states = [StateType("link", "a_tip", "pos", "world")]
        elif family == "pose":
            states = [StateType("link", "a_tip", key, "world") for key in ("pos", "rot")]
        elif family == "kinematics":
            states = [StateType("link", "a_tip", key, "world") for key in ("pos", "vel", "acc")]
        elif family == "torque":
            states = [StateType("total_joint", "total_joint", "torque")]
        else:
            states = [StateType("link", "a_tip", "acc", "world"),
                      StateType("joint", "b_payload_fixed", "force" if order == 3 else "force_diff3", "world"),
                      StateType("link", "a_tip", "pos", "world")]
        objects = {backend: (Kots.from_json_data(model_data, order=order, backend=backend) if model_data
                             else Kots.from_urdf_file(str(MODEL), order=order, backend=backend))
                   for backend in ("numpy", "rust")}
        dof = objects["numpy"].dof()
        motions = [rng.normal(scale=.2, size=shape + (dof*order,)) for _ in range(2)]

        def prepare(k, motion):
            k.import_motions(motion)
            if family in ("position", "pose", "kinematics"):
                k.kinematics()
            else:
                k.dynamics(gravity=gravity)

        for k in objects.values():
            prepare(k, motions[0])
        reference = objects["numpy"].jacobian(states)
        direction = rng.normal(size=shape + (reference.shape[-1],))
        weight = rng.normal(size=shape + (reference.shape[-2],))

        def operation(k, name):
            if name == "dense":
                return k.jacobian(states)
            if name == "jvp":
                return k.jacobian_mul(states, direction)
            return k.jacobian_transpose_mul(states, weight)

        def forbidden(*args, **kwargs):
            raise AssertionError("unexpected cross-backend fallback")

        for scope in ("state_ready", "including_state"):
            for name in ("dense", "jvp", "vjp"):
                times, outputs = {}, {}
                for backend, k in objects.items():
                    prepare(k, motions[0])
                    with ExitStack() as stack:
                        if backend == "rust":
                            for entry in ("outward_jacobian", "outward_jacobian_matvec", "outward_jacobian_matmul_rhs",
                                          "outward_jacobian_transpose_matvec", "outward_jacobian_transpose_matmul_rhs"):
                                stack.enter_context(patch.object(outward_api, entry, forbidden))
                        else:
                            stack.enter_context(patch.object(k, "_rust_compiled_robot", forbidden))
                        outputs[backend] = np.asarray(operation(k, name))
                        index = [0]
                        def fn():
                            if scope == "including_state":
                                # Change actual values, not only the revision, to invalidate primal caches.
                                index[0] = 1-index[0]
                                prepare(k, motions[index[0]])
                            return operation(k, name)
                        times[backend] = measure(fn, warmup, samples, calls)
                delta = outputs["numpy"]-outputs["rust"]
                np.testing.assert_allclose(outputs["numpy"], outputs["rust"], rtol=2e-10, atol=2e-10)
                if name != "dense":
                    expected = (reference @ direction[..., None])[..., 0] if name == "jvp" else (reference.swapaxes(-1, -2) @ weight[..., None])[..., 0]
                    np.testing.assert_allclose(outputs["numpy"], expected, rtol=2e-10, atol=2e-10)
                row = dict(family=family, order=order, batch_shape=list(shape), jacobian_shape=list(reference.shape),
                           scope=scope, operation=name, timings=times,
                           rust_speedup=times["numpy"]["median_ms"]/times["rust"]["median_ms"],
                           max_abs=float(np.max(np.abs(delta))),
                           relative_frobenius=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(outputs["numpy"].ravel()), 1e-30)))
                rows.append(row)
        print(f"Finished {family}, order={order}, batch={shape}", flush=True)
    import robokots._rust_core as native
    result = dict(environment=dict(platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                                   rust_extension_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()),
                  model=model_name, model_data=model_data, dof=dof, seed=seed, dtype="float64", gravity=gravity,
                  warmup=warmup, samples=samples, calls_per_sample=calls,
                  notes="Same-build analytical backends; no JAX or numerical differentiation. Rust NumPy fallbacks and NumPy native dispatch blocked. Input/output boundary costs included. Model construction excluded. state_ready reuses current state and warmed derivative workspaces; including_state alternates two motions and includes import + state calculation + derivative. First_ms is first timed call, not cold compilation. Milliseconds per entire call (batch time is for six samples).",
                  rows=rows)
    target = output or ROOT / "developer/benchmarks/results/numpy_rust_jacobian_compare.json"
    target.write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# NumPy / Rust analytical Jacobian comparison", "", result["notes"], "",
             f"Model: {result['model']} ({dof} DOF). Float64; seed {seed}; gravity {gravity}.",
             f"Warmup {warmup}; {samples} samples × {calls} calls; median milliseconds.", "",
             "| Family | Order | Batch | Scope | Operation | NumPy ms | Rust ms | Speedup |",
             "|---|---:|---|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['family']} | {r['order']} | {r['batch_shape']} | {r['scope']} | {r['operation']} | {r['timings']['numpy']['median_ms']:.6f} | {r['timings']['rust']['median_ms']:.6f} | {r['rust_speedup']:.1f}× |")
    lines += ["", f"Maximum absolute difference: {max(r['max_abs'] for r in rows):.6g}.",
              f"Maximum relative Frobenius error: {max(r['relative_frobenius'] for r in rows):.6g}.",
              "Full timing samples, first calls, environment, and shapes are in the adjacent JSON."]
    target.with_suffix(".md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", choices=("position", "pose", "kinematics", "torque", "mixed"))
    parser.add_argument("--serial-dof", type=int, help="Synthetic serial model; requires a non-mixed --case")
    args = parser.parse_args()
    main(args.output, args.case, args.serial_dof)
