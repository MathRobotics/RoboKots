"""Compare two release extensions in fresh processes through production APIs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter_ns

from .state_cache_scope import ROOT, RESULTS, load_probe


def worker(args):
    load_probe(args.extension)
    import numpy as np
    from robokots.kots import Kots, StateType
    from robokots.core.robot import RobotStruct
    from .common import build_model
    rng = np.random.default_rng(917)
    rows = []
    for dof in (16, 64):
        for batch in (1, 8):
            for order in (3, 4, 5, 6, 8):
                k = Kots(RobotStruct.from_dict(build_model(dof, 'humanoid')), order=order, dim=3)
                raw = k._rust_compiled_robot().create_batch_outward_data(order, batch)
                motion = rng.normal(scale=.2, size=(batch, dof * order))
                for nonzero in (False, True):
                    gravity = np.array([.3, -.4, -9.81] if nonzero else [0, 0, 0], dtype=float)
                    def public():
                        # Import invalidates cached state on every evaluation.
                        k.import_motions(motion)
                        k.dynamics(backend='rust', gravity=gravity)
                    calls = {'raw': lambda: raw.compute_dynamics(motion, gravity), 'public': public}
                    def time_call(call, loops):
                        start = perf_counter_ns()
                        for _ in range(loops): call()
                        return (perf_counter_ns() - start) / loops / 1000
                    first = {name: time_call(call, 1) for name, call in calls.items()}
                    for _ in range(args.warmup):
                        for call in calls.values(): call()
                    samples = {name: [] for name in calls}
                    for i in range(args.repeats):
                        for name in (('raw', 'public') if i % 2 else ('public', 'raw')):
                            samples[name].append(time_call(calls[name], args.loops))
                    values = []
                    for family, count in [('momentum', order-1), ('force', order-2)]:
                        for owner, names in [('link', k.link_name_list()), ('joint', k.joint_name_list())]:
                            for name in names:
                                for n in range(count):
                                    key = family if n == 0 else f'{family}_diff{n}'
                                    values.extend(np.asarray(k.state_info(StateType(owner, name, key))).reshape(-1))
                    rows.append(dict(dof=dof, batch=batch, order=order, gravity=gravity.tolist(), first_us=first, samples_us=samples, values=values))
    args.output.write_text(json.dumps(dict(numpy=np.__version__, rows=rows)))


def run(args):
    import numpy as np
    results = {'baseline': [], 'optimized': []}
    with tempfile.TemporaryDirectory(prefix='robokots-production-') as directory:
        for round_id in range(args.rounds):
            names = ('baseline', 'optimized') if round_id % 2 == 0 else ('optimized', 'baseline')
            for name in names:
                output = Path(directory) / f'{name}-{round_id}.json'
                subprocess.run([sys.executable, '-m', 'developer.benchmarks.high_order_production', '--extension', str(getattr(args, name)), '--output', str(output), '--warmup', str(args.warmup), '--repeats', str(args.repeats), '--loops', str(args.loops)], check=True, cwd=ROOT)
                results[name].append(json.loads(output.read_text()))
                print(f'Completed round {round_id+1}/{args.rounds}: {name}', flush=True)
    rows = []
    for i, original in enumerate(results['baseline'][0]['rows']):
        row = {key: original[key] for key in ('dof', 'batch', 'order', 'gravity')}
        ref = np.asarray(original['values'])
        row['variants'] = {}
        for name, rounds in results.items():
            error = []
            for result in rounds:
                actual = np.asarray(result['rows'][i]['values'])
                np.testing.assert_allclose(actual, ref, atol=5e-8, rtol=5e-9)
                error.append(dict(max_abs=float(np.max(np.abs(actual-ref))), relative_fro=float(np.linalg.norm(actual-ref)/max(np.linalg.norm(ref), 1e-30))))
            samples = {phase: [x for r in rounds for x in r['rows'][i]['samples_us'][phase]] for phase in ('raw', 'public')}
            row['variants'][name] = dict(samples_us=samples, median_us={p: float(np.median(v)) for p,v in samples.items()}, first_us=[r['rows'][i]['first_us'] for r in rounds], error=error)
        rows.append(row)
    metadata = dict(measured_at=datetime.now(timezone.utc).isoformat(), platform=platform.platform(), python=sys.version, numpy=np.__version__, rustc=subprocess.check_output(['rustc', '--version'], text=True).strip(), seed=917, warmup=args.warmup, repeats_per_round=args.repeats, rounds=args.rounds, loops=args.loops, dtype='float64', binary_sha256={name: hashlib.sha256(getattr(args,name).read_bytes()).hexdigest() for name in results}, source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'robokots/_rust/src').glob('*.rs') if not p.name.startswith('._')})
    baseline_metadata = args.baseline.with_name('baseline_metadata.json')
    if baseline_metadata.exists(): metadata['baseline_build'] = json.loads(baseline_metadata.read_text())
    RESULTS.mkdir(exist_ok=True)
    (RESULTS/'high_order_production.json').write_text(json.dumps(dict(metadata=metadata, rows=rows), indent=2)+'\n')
    lines = ['# High-order production dynamics comparison', '', f"Environment: {metadata['platform']}; Python {sys.version.split()[0]}, NumPy {np.__version__}, {metadata['rustc']}. Seed 917, float64.", f'Fresh processes, alternating binary order across {args.rounds} rounds; {args.warmup} warmups, {args.repeats} samples per round, {args.loops} evaluations per sample. Median microseconds per batch. First-call times, raw samples, binary and source hashes are in JSON.', '', 'Raw = persistent Rust batch workspace compute_dynamics, including Python binding cost. Public = import_motions + dynamics(backend="rust"), including validation, conversion, cache invalidation and state update. Every timed call recomputes dynamics; no cache-hit timings. Model/workspace construction and state extraction are outside timing. This measures state computation, not Jacobian/JVP/VJP evaluation.', '', '|DOF|Batch|Order|Gravity|Raw before|Raw after|Public before|Public after|Public speedup|', '|---:|---:|---:|:---:|---:|---:|---:|---:|---:|']
    for r in rows:
        b=r['variants']['baseline']['median_us']; a=r['variants']['optimized']['median_us']
        lines.append(f"|{r['dof']}|{r['batch']}|{r['order']}|{'nonzero' if any(r['gravity']) else 'zero'}|{b['raw']:.2f}|{a['raw']:.2f}|{b['public']:.2f}|{a['public']:.2f}|{b['public']/a['public']:.2f}x|")
    errors=[e for r in rows for e in r['variants']['optimized']['error']]
    lines += ['', f"All local link/joint momentum and force derivatives agree with the previous binary: maximum absolute difference {max(e['max_abs'] for e in errors):.6g}; maximum relative Frobenius error {max(e['relative_fro'] for e in errors):.6g}.", '', 'Production changes: order >=4 fixed/revolute kinematics directly propagates spatial velocity derivatives; gravity transport reuses blocks already built for momentum. Order-3 zero-gravity specialized dispatch is preserved. No persistent state fields are added. Temporary rotation scratch is 72*(order-1) bytes per evaluation, reused across joints (216 bytes at order 4, 504 bytes at order 8). Results are environment-dependent.', '']
    (RESULTS/'high_order_production.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--optimized', type=Path)
    parser.add_argument('--extension', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--loops', type=int, default=5)
    args = parser.parse_args()
    if args.extension and args.output: worker(args)
    elif args.baseline and args.optimized: run(args)
    else: parser.error('provide --baseline/--optimized, or --extension/--output')
