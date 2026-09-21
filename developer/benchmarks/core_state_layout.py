"""Compare state-layout refactors using identical public API workloads.

Run before and after a refactor with --output; add --compare BEFORE.json after.
Uses NumPy/Rust only (no JAX/JIT timing). No numerical kernels are substituted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from importlib.metadata import version

import numpy as np
from robokots.kots import Kots, StateType

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'tests/test_model/sample_robot.json'
SEED = 71
ORDER = 4
GRAVITY = (0.0, 0.0, -9.81)


def measure(fn, warmup, samples, calls):
    start = time.perf_counter_ns()
    fn()
    first = (time.perf_counter_ns() - start) / 1e6
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        for _ in range(calls):
            fn()
        timings.append((time.perf_counter_ns() - start) / calls / 1e6)
    return dict(first_ms=first, median_ms=statistics.median(timings), samples_ms=timings)


def run(args):
    results, values = {}, {}
    for backend in ('numpy', 'rust'):
        for shape in ((), (2, 3)):
            k = Kots.from_json_file(str(MODEL), order=ORDER)
            motion = np.random.default_rng(SEED).normal(scale=.2, size=shape + (k.dof()*ORDER,))
            k.import_motions(motion)
            name = f'{backend}/' + ('batch_2x3' if shape else 'single')
            specs = [StateType('link', k.link_name_list()[-1], quantity)
                     for quantity in ('pos', 'vel', 'force')]

            def dynamics():
                k.import_motions(motion)
                k.dynamics(backend=backend, gravity=GRAVITY)

            results[name + '/import_and_dynamics'] = measure(dynamics, args.warmup, args.samples, args.calls)
            results[name + '/read_values'] = measure(lambda: k.state_info_list(specs), args.warmup, args.samples, args.calls)
            results[name + '/export'] = measure(k.to_state_dict, args.warmup, args.samples, args.calls)
            values[name] = np.asarray(k.state_info_list(specs)).tolist()
            if not shape and backend == 'numpy':
                k.update_state(is_dynamics=True, backend=backend)
                results[name + '/cache_hit'] = measure(
                    lambda: k.update_state(is_dynamics=True, backend=backend),
                    args.warmup, args.samples, args.calls)

    # Exercise the list-of-states fallback explicitly, without timing its setup.
    k = Kots.from_json_file(str(MODEL), order=ORDER)
    motions = np.random.default_rng(SEED).normal(scale=.2, size=(6, k.dof()*ORDER))
    states = []
    for motion in motions:
        k.import_motions(motion)
        states.append(k.dynamics(backend='numpy', gravity=GRAVITY))
    k._set_batch_states(states, (2, 3))
    specs = [StateType('link', k.link_name_list()[-1], q) for q in ('pos', 'vel', 'force')]
    results['list_batch/read_values'] = measure(lambda: k.state_info_list(specs), args.warmup, args.samples, args.calls)
    results['list_batch/read_parts'] = measure(lambda: k.state_info_list(specs, list_output=True), args.warmup, args.samples, args.calls)
    results['list_batch/export'] = measure(k.to_state_dict, args.warmup, args.samples, args.calls)
    values['list_batch'] = np.asarray(k.state_info_list(specs)).tolist()
    return dict(
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         numpy=version('numpy'), mathrobo=version('mathrobo'),
                         threads={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
                         git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                         git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)),
        workload=dict(model=str(MODEL.relative_to(ROOT)), model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                      seed=SEED, motion_order=ORDER, gravity=list(GRAVITY), dtype='float64',
                      warmup=args.warmup, samples=args.samples, calls_per_sample=args.calls,
                      note='Import/conversion included only in import_and_dynamics. Reads/export use computed states. '
                           'First timing is first invocation of each workload, not process import time. '
                           'List batch is explicitly assembled from NumPy scalar states. No JAX/JIT measurements.'),
        timings=results, values=values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--calls', type=int, default=5)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = run(args)
    if args.compare:
        before = json.loads(args.compare.read_text())
        assert before['workload'] == report['workload'], 'Workload mismatch'
        comparison = {}
        for name, current in report['values'].items():
            a, b = np.asarray(before['values'][name]), np.asarray(current)
            diff = b - a
            comparison[name] = dict(max_abs=float(np.max(np.abs(diff))),
                                    relative_frobenius=float(np.linalg.norm(diff) / max(np.linalg.norm(a), np.finfo(float).tiny)))
            np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12)
        report['accuracy_vs_before'] = comparison
        lines = ['# Core state layout comparison', '',
                 'Same model, float64 motion order 4, seed 71, world gravity [0, 0, -9.81].',
                 f'{args.warmup} warmups; {args.samples} samples × {args.calls} calls; medians in ms/call.',
                 'Before/after run in separate processes. Small timing changes may be measurement noise.', '',
                 '| Workload | Before ms | After ms | After / before |', '| --- | ---: | ---: | ---: |']
        for name, timing in report['timings'].items():
            old = before['timings'][name]['median_ms']
            new = timing['median_ms']
            lines.append(f'| {name} | {old:.6f} | {new:.6f} | {new/old:.3f} |')
        lines += ['', 'Maximum absolute differences and relative Frobenius errors:', '',
                  '| Output | Max abs | Relative Frobenius |', '| --- | ---: | ---: |']
        for name, errors in comparison.items():
            lines.append(f"| {name} | {errors['max_abs']:.3g} | {errors['relative_frobenius']:.3g} |")
        lines += ['', report['workload']['note'], '', 'Environment and individual timings are recorded in the JSON files.', '']
        args.output.with_suffix('.md').write_text('\n'.join(lines))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
