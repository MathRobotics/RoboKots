"""Before/after kernel-layout timings; dense Jacobian, JVP and VJP separately.

Run with --output before.json, then --output after.json --compare before.json.
State building is excluded from derivative timings. NumPy/Rust float64 only;
there is no JAX/JIT measurement or numerical-difference reference in this probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from importlib.metadata import version
from unittest.mock import patch

import numpy as np
import robokots
from robokots.kots import Kots, StateType
from .core_state_layout import measure

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'tests/test_model/sample_robot.json'


def error(actual, reference):
    a, b = np.asarray(actual), np.asarray(reference)
    diff = a - b
    return dict(max_abs=float(np.max(np.abs(diff))),
                relative_frobenius=float(np.linalg.norm(diff) / max(np.linalg.norm(b), np.finfo(float).tiny)))


def run(args):
    timings, outputs, products = {}, {}, {}
    for backend in ('numpy', 'rust'):
        for shape in ((), (2,)):
            rng = np.random.default_rng(71)
            k = Kots.from_json_file(str(MODEL), order=4)
            x = rng.normal(scale=.2, size=shape + (k.dof()*4,))
            name = backend + ('/batch2' if shape else '/single')
            states = [StateType('link', k.link_name_list()[-1], 'vel'),
                      StateType('link', k.link_name_list()[-1], 'force_diff1'),
                      StateType('joint', k.joint_name_list()[-1], 'torque_diff1')]
            def build():
                k.import_motions(x)
                k.dynamics(backend=backend, gravity=(.2, -.3, -9.81))
            timings[name + '/import_and_dynamics'] = measure(build, args.warmup, args.samples, args.calls)
            # Materialized here only to define shapes and independently check products.
            jac = k.jacobian(states)
            v = rng.normal(size=shape + (jac.shape[-1],))
            w = rng.normal(size=shape + (jac.shape[-2],))
            operations = {'dense': lambda: k.jacobian(states),
                          'jvp': lambda: k.jacobian_mul(states, v),
                          'vjp': lambda: k.jacobian_transpose_mul(states, w)}
            # Check dispatch outside timing: local mixed outputs must use products.
            with patch.object(k, '_jacobian_from_state', side_effect=AssertionError('Dense fallback')):
                operations['jvp']()
                operations['vjp']()
            for operation, fn in operations.items():
                key = name + '/' + operation
                timings[key] = measure(fn, args.warmup, args.samples, args.calls)
                outputs[key] = np.asarray(fn()).tolist()
            outputs[name + '/state'] = np.asarray(k.state_info_list(states)).tolist()
            for operation, expected in (
                ('jvp', (jac @ v[..., None])[..., 0]),
                ('vjp', (np.swapaxes(jac, -1, -2) @ w[..., None])[..., 0]),
            ):
                key = name + '/' + operation
                products[key] = error(outputs[key], expected)
                np.testing.assert_allclose(outputs[key], expected, atol=1e-9, rtol=1e-9)
    return dict(
        workload=dict(model=str(MODEL.relative_to(ROOT)), model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                      seed=71, motion_order=4, gravity=[.2, -.3, -9.81], dtype='float64',
                      states=['last link vel local (default)', 'last link force_diff1 local (default)', 'last joint torque_diff1 local (default)'],
                      warmup=args.warmup, samples=args.samples, calls_per_sample=args.calls,
                      note='State computation and input conversion included only in import_and_dynamics. '
                           'Dense/JVP/VJP start with computed states and use public dispatch; internal scalar-state fallback work is included. '
                           'Products are checked with the dense facade path disabled outside timing. '
                           'Rust labels identify state storage: dense/JVP use Python analytic derivatives over Rust views; VJP uses Rust direct kernels for this selection. '
                           'First calls follow state setup and product-shape discovery; import/JIT costs are not measured.'),
        environment=dict(platform=platform.platform(), python=platform.python_version(),
                         robokots_source=str(Path(robokots.__file__).resolve().parent),
                         numpy=version('numpy'), mathrobo=version('mathrobo'),
                         threads={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
                         git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                         git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)),
        timings=timings, outputs=outputs, direct_vs_dense=products)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--calls', type=int, default=5)
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.compare:
        before = json.loads(args.compare.read_text())
        assert before['workload'] == report['workload'], 'Workload mismatch'
        report['accuracy_vs_before'] = {key: error(value, before['outputs'][key]) for key, value in report['outputs'].items()}
        for key, value in report['outputs'].items():
            np.testing.assert_allclose(value, before['outputs'][key], atol=1e-12, rtol=1e-12)
        lines = ['# Kernel layout comparison', '',
                 f'{args.warmup} warmups, {args.samples} samples × {args.calls} calls. Medians in ms/call.',
                 'Before/after are separate processes; small differences may be measurement noise.', '',
                 '| Workload | Before ms | After ms | After / before |', '| --- | ---: | ---: | ---: |']
        for key, timing in report['timings'].items():
            a, b = before['timings'][key]['median_ms'], timing['median_ms']
            lines.append(f'| {key} | {a:.6f} | {b:.6f} | {b/a:.3f} |')
        lines += ['', '| Output | Max abs difference | Relative Frobenius error |', '| --- | ---: | ---: |']
        for key, value in report['accuracy_vs_before'].items():
            lines.append(f"| {key} | {value['max_abs']:.3g} | {value['relative_frobenius']:.3g} |")
        lines += ['', report['workload']['note'], '',
                  'Model, gravity, state selections, seed, environment, raw timings, outputs and direct-vs-dense checks are in the JSON files.', '']
        args.output.with_suffix('.md').write_text('\n'.join(lines))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
