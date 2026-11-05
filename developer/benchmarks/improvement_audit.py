"""Read-only improvement audit; prototypes are local and do not patch RoboKots.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m
developer.benchmarks.improvement_audit
Outputs JSON to stdout. Timing comparisons alternate ABBA after three warmups.
This historical probe targets the pre-fix cc5eede implementation. Some probes
intentionally exercise behavior now rejected by validation; reproduce on that
implementation rather than treating this script as a current regression suite.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import time
from importlib.metadata import version
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robokots.kots import Kots, StateType
from robokots.core.state.protocol import OutwardDataView
from robokots.api.state_cache import StateCache
from robokots.core.state.batch import StateBatch
from robokots.state_io.dictionary import export_state_dict
from robokots.state_io.jsonl import iter_jsonl_rows, make_jsonl_row
from developer.benchmarks.common import build_model


def sample(order=3, batch=()):
    k = Kots.from_json_file("tests/test_model/sample_robot.json", order=order)
    x = np.random.default_rng(71).normal(scale=.2, size=batch + (k.dof()*order,))
    k.import_motions(x)
    return k, x


def outcome(fn):
    try:
        value = fn()
        return dict(ok=True, shape=list(np.shape(value)))
    except Exception as exc:
        return dict(ok=False, error=type(exc).__name__, message=str(exc))


def compare(a, b):
    samples = [[], []]
    first = []
    for fn in (a, b):
        start = time.perf_counter_ns()
        fn()
        first.append((time.perf_counter_ns()-start)/1e6)
        for _ in range(3):
            fn()
    for _ in range(15):
        for i in (0, 1, 1, 0):
            start = time.perf_counter_ns()
            (a, b)[i]()
            samples[i].append((time.perf_counter_ns()-start)/1e6)
    med = [statistics.median(s) for s in samples]
    return dict(first_ms=first, samples_ms=samples, median_ms=med,
                ratio=med[0]/med[1])


def checks():
    result = {}
    for backend in ("numpy", "rust"):
        row = {}
        k, x = sample()
        k.update_state(backend=backend)
        st = StateType("link", k.link_name_list()[-1], "pos")
        x[0] += .7
        k.update_state(backend=backend)
        stale = k.state_info(st).copy()
        k.import_motions(x.copy())
        k.update_state(backend=backend)
        row["input_alias_stale_max_abs"] = float(np.max(abs(stale-k.state_info(st))))
        # Copying at the boundary prevents caller-owned input mutation.
        k, x = sample()
        k.import_motions(x.copy())
        original = k.motion().copy()
        x[0] += .7
        row["copy_input_prototype_max_abs"] = float(np.max(abs(original-k.motion())))
        k.dynamics(backend=backend)
        row["protocol_matches"] = isinstance(k.outward_state_, OutwardDataView)
        states = [st, StateType("link", st.owner_name, "vel")]
        row["mixed_query"] = outcome(lambda: k.state_info_list(states))
        parts = k.state_info_list(states, list_output=True)
        row["concatenate_prototype_shape"] = list(np.concatenate([np.asarray(p).reshape(-1) for p in parts]).shape)
        row["empty_query"] = outcome(lambda: k.state_info_list([]))
        row["query_alias_max_abs"] = {}
        row["query_copy_prototype_max_abs"] = {}
        for quantity in ("vel", "force", "torque"):
            k.dynamics(backend=backend)
            owner = "joint" if quantity == "torque" else "link"
            name = k.joint_name_list()[-1] if owner == "joint" else st.owner_name
            query = StateType(owner, name, quantity)
            value = k.state_info(query)
            before = value.copy()
            value[...] = 123
            row["query_alias_max_abs"][quantity] = float(np.max(abs(k.state_info(query)-before)))
            baseline = k.state_info(query).copy()
            copied = k.state_info(query).copy()
            copied[...] = -456
            row["query_copy_prototype_max_abs"][quantity] = float(np.max(abs(k.state_info(query)-baseline)))
        k, x = sample()
        old = k.dynamics(backend=backend, gravity=[0, 0, -9.81])
        matrix = old.cmtm("link", st.owner_name).elem_mat().copy()
        k.import_motions(x*.3)
        k.dynamics(backend=backend, gravity=[0, 0, -9.81])
        row["old_state_handle_change"] = float(np.max(abs(matrix-old.cmtm("link", st.owner_name).elem_mat())))
        row["failed_dynamics"] = outcome(lambda: k.dynamics(backend="unsupported", gravity=[0, 0, 1]))
        row["gravity_after_failure"] = k.gravity_.tolist()
        row["state_gravity_after_failure"] = k.outward_state_.gravity.tolist()
        k.kinematics(backend=backend)
        row["dynamics_keys_after_kinematics"] = sum("force" in s or "torque" in s for s in k.to_state_dict())
        result[backend] = row

    calls = []
    def broken(x, **kwargs):
        calls.append(kwargs)
        raise TypeError("internal builder error")
    result["builder_error"] = outcome(lambda: StateCache(broken).update_if_needed(
        SimpleNamespace(revision=1, get=lambda: np.zeros(1))))
    result["builder_call_count"] = len(calls)
    # Fixed callable contract propagates the original error after one call.
    calls.clear()
    outcome(lambda: broken(np.zeros(1), time=None, required=None))
    result["single_signature_prototype_calls"] = len(calls)
    result["jsonl_short_times_rows"] = len(list(iter_jsonl_rows([{}, {}, {}], times=[0.])))
    result["jsonl_both_axes"] = list(iter_jsonl_rows([{}], times=[.1], steps=[3]))
    result["jsonl_reserved_overwrite"] = make_jsonl_row({}, step=2, meta={"step":8,"schema_version":999})
    result["strict_zip_prototype"] = outcome(lambda: list(zip([{}, {}, {}], [0.], strict=True)))
    invalid = StateBatch.from_states([object()], (2, 3))
    result["invalid_batch_accepted"] = dict(count=len(invalid.outward_states), shape=invalid.batch_shape)
    result["batch_exports"] = {}
    for model in ("sample_robot", "soft_rod"):
        k = Kots.from_json_file(f"tests/test_model/{model}.json", order=3)
        k.import_motions(np.zeros((2, 1, k.dof()*3)))
        k.kinematics()
        exported = k.to_state_dict()
        result["batch_exports"][model] = dict(type=type(exported).__name__, batch_shape=k.batch_shape_)
        if isinstance(exported, list):
            packed = {key: np.stack([s[key] for s in exported]).reshape(k.batch_shape_ + np.asarray(exported[0][key]).shape)
                      for key in exported[0]}
            result["batch_exports"][model]["stack_prototype_first_shape"] = list(next(iter(packed.values())).shape)
    k, _ = sample()
    for n in range(1, 9):
        k.import_motions(np.zeros((n, k.dof()*3)))
        k.kinematics(backend="rust")
    result["rust_workspace_entries_for_8_shapes"] = len(k._rust_outward_data_cache_)
    # Simulate a two-entry bound; revisit an evicted shape and verify recomputation.
    k, _ = sample()
    reference = None
    for n in (1,2,3,4,5,6,7,8,1):
        k.import_motions(np.zeros((n, k.dof()*3)))
        k.kinematics(backend="rust")
        while len(k._rust_outward_data_cache_) > 2:
            key = next(iter(k._rust_outward_data_cache_))
            del k._rust_outward_data_cache_[key]
            k._rust_outward_data_cache_state_.pop(key, None)
        if n == 1:
            exported = k.to_state_dict()
            if reference is None:
                reference = exported
            else:
                for key in reference:
                    np.testing.assert_array_equal(reference[key], exported[key])
    result["bounded_workspace_prototype"] = dict(entries=len(k._rust_outward_data_cache_), revisited_evicted_shape_equal=True)

    from robokots.contrib.polars.state_table import RobotState
    k, _ = sample()
    k.kinematics()
    k.set_state_df()
    df = k.state_df()
    names = k.link_name_list()
    result["polars_velocity_trajectory"] = outcome(lambda: RobotState.state_vecs_traj(df, names, "link", "vel"))
    result["polars_stack_prototype_shape"] = list(np.stack([
        np.asarray(df[f"{name}_link_vel"].to_list()) for name in names]).shape)
    k.dynamics()
    k.set_state_df()
    result["polars_momentum_columns"] = sum("momentum" in key for key in k.state_df().columns)
    result["export_momentum_keys"] = sum("momentum" in key for key in k.to_state_dict())

    from robokots import outward as outward_api
    original = outward_api.build_kinematics_outward_state
    calls = []
    def batch_fault(robot, motion, *args, **kwargs):
        calls.append(list(np.shape(motion)))
        if np.ndim(motion) > 1:
            raise RuntimeError("injected unexpected batched kernel error")
        return original(robot, motion, *args, **kwargs)
    k, _ = sample(batch=(2,))
    with patch.object(outward_api, "build_kinematics_outward_state", batch_fault):
        result["unexpected_batch_error_swallowed"] = outcome(lambda: k.kinematics())
    result["fallback_call_shapes"] = calls
    result["numpy_repeated_update_builder_calls"] = {}
    for batch in ((), (2,)):
        k, _ = sample(batch=batch)
        with patch.object(outward_api, "build_kinematics_outward_state", wraps=original) as spy:
            k.update_state()
            k.update_state()
            result["numpy_repeated_update_builder_calls"][str(batch)] = spy.call_count
    return result


def timings():
    rows = []
    for dof in (3, 16):
        for backend in ("numpy", "rust"):
            k = Kots.from_json_file("tests/test_model/sample_robot.json", order=5) if dof == 3 else Kots.from_json_data(build_model(dof, "humanoid"), order=5)
            x = np.random.default_rng(71).normal(scale=.2, size=k.dof()*5)
            k.import_motions(x)
            k.dynamics(backend=backend, gravity=[.2,-.3,-9.81])
            name = k.joint_name_list()[-1]
            full = k.to_state_dict()
            # Obtain the canonical key rather than assuming the serialized layout.
            from robokots.core.state.spec import state_dict_key
            key = state_dict_key("joint", name, "torque")
            def selected():
                return {key: k.outward_state_.quantity_series("joint", name, "torque")[...,0,:].copy()}
            np.testing.assert_array_equal(selected()[key], full[key])
            row = dict(operation="full_export_vs_selected_torque", dof=dof, backend=backend,
                       full_keys=len(full), selected_keys=1,
                       full_array_bytes=sum(np.asarray(v).nbytes for v in full.values()),
                       selected_array_bytes=selected()[key].nbytes,
                       max_abs_error=0., relative_frobenius_error=0., **compare(k.to_state_dict, selected))
            rows.append(row)
            # Same model, motion, order, gravity; repeated reads with no revision changes.
            k.update_state(is_dynamics=True, backend=backend)
            cached_values = k.to_state_dict()
            k.dynamics(backend=backend, gravity=[.2,-.3,-9.81])
            fresh_values = k.to_state_dict()
            for key in cached_values:
                np.testing.assert_array_equal(cached_values[key], fresh_values[key])
            rows.append(dict(operation="dynamics_vs_fresh_update", dof=dof, backend=backend,
                             max_abs_error=0., relative_frobenius_error=0.,
                             **compare(lambda:k.dynamics(backend=backend,gravity=[.2,-.3,-9.81]),
                                       lambda:k.update_state(is_dynamics=True,backend=backend))))

    import polars as pl
    from robokots.contrib.polars.state_table import RobotDF
    names = [f"quantity{i}" for i in range(10)]
    values = [{name: [float(i), 2., 3.] for name in names} for i in range(100)]
    schema = {name:pl.List(pl.Float64) for name in names}
    def incremental():
        table = RobotDF(names)
        for value in values:
            table.add_row(value)
        return table.df
    def bulk():
        return pl.DataFrame(values, schema=schema)
    assert incremental().equals(bulk())
    rows.append(dict(operation="polars_100_rows_incremental_vs_bulk", equal=True, **compare(incremental,bulk)))
    return rows


def jax_prototypes():
    import jax
    import jax.numpy as jnp
    from robokots.outward.diff.dynamics_jax import dynamics_state_vector_jax
    jax.config.update("jax_enable_x64", True)
    k, motion = sample()
    gravity = np.array([.2, -.3, -9.81])
    states = [StateType("joint", k.joint_name_list()[-1], "torque")]
    def value(x):
        return dynamics_state_vector_jax(k.robot_, x, states, 3, gravity)
    forward = jax.jit(jax.jacfwd(value))
    reverse = jax.jit(jax.jacrev(value))
    jvp = jax.jit(lambda x, v: jax.jvp(value, (x,), (v,))[1])
    vjp = jax.jit(lambda x, w: jax.vjp(value, x)[1](w)[0])
    x = jnp.asarray(motion)
    v = jnp.asarray(np.random.default_rng(72).normal(size=motion.shape))
    w = jnp.ones(1)
    first = {}
    for name, run in (("forward", lambda: forward(x)), ("reverse", lambda: reverse(x)),
                      ("jvp",lambda:jvp(x,v)), ("vjp",lambda:vjp(x,w))):
        start = time.perf_counter_ns()
        run().block_until_ready()
        first[name] = (time.perf_counter_ns()-start)/1e6
    errors = []
    for scale in (1., .4):
        k.import_motions(motion*scale)
        k.dynamics(gravity=gravity)
        analytic = k.jacobian(states)
        actual = np.asarray(forward(x*scale).block_until_ready())
        np.testing.assert_allclose(actual, analytic, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(reverse(x*scale), actual, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(jvp(x*scale,v), actual@v, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(vjp(x*scale,w), actual.T@w, atol=1e-10, rtol=1e-10)
        errors.append(dict(max_abs=float(np.max(abs(actual-analytic))),
                           relative_frobenius=float(np.linalg.norm(actual-analytic)/max(np.linalg.norm(analytic),1e-300))))
    return dict(jax=version("jax"), shape=list(forward(x).shape),
                first_compile_and_execute_ms=first, analytic_errors_two_inputs=errors,
                forward_vs_reverse=compare(lambda:np.asarray(forward(x).block_until_ready()),
                                           lambda:np.asarray(reverse(x).block_until_ready())),
                direct_products_validated=True,
                note="3 DOF, order 3, one joint torque; JIT timings include synchronization and NumPy output conversion, but not state calculation in Kots")


if __name__ == "__main__":
    print(json.dumps(dict(metadata=dict(commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        python=platform.python_version(), numpy=np.__version__, mathrobo=version("mathrobo"),
        platform=platform.platform(), timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        thread_environment={key:os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS")},
        seed=71, dtype="float64", warmup=3, repeats=30, ordering="ABBA",
        timing_scope="same-process microbenchmarks; numerical differentiation and model initialization excluded; separate JAX section includes compilation and synchronization"),
        checks=checks(), timings=timings(), jax=jax_prototypes()), indent=2))
