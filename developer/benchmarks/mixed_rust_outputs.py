"""Compare local mixed-output Rust support before/after a release rebuild.

Uses the fixed workload in kernel_layout: order 4, float64, seed 71, sample
robot, local velocity/force_diff1/torque_diff1, nonzero gravity, scalar/batch2.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .kernel_layout import run, error
import robokots._rust_core as extension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--calls', type=int, default=5)
    args = parser.parse_args()
    report = run(args)
    report['rust_vs_numpy'] = {}
    for key, value in report['outputs'].items():
        if key.startswith('rust/'):
            reference = report['outputs'][key.replace('rust/', 'numpy/', 1)]
            report['rust_vs_numpy'][key] = error(value, reference)
            np.testing.assert_allclose(value, reference, atol=1e-10, rtol=1e-10)
    report['workload']['note'] = (
        'Public dense/JVP/VJP start with computed states; any internal recurrence recomputation is timed. '
        'Import/conversion plus state building is measured separately. NumPy is a reference, not an exact solution. '
        'Before: Rust state, Python dense/JVP, composed Rust VJP. After: all mixed derivatives use the selected Rust recurrence. '
        'No JAX/JIT timing. First calls follow shape discovery, not cold process startup.')
    report['environment']['extension_sha256'] = hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()
    report['environment']['extension_path'] = extension.__file__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.compare:
        before = json.loads(args.compare.read_text())
        assert before['workload'] == report['workload']
        report['accuracy_vs_before'] = {key: error(value, before['outputs'][key]) for key, value in report['outputs'].items()}
        for key, value in report['outputs'].items():
            np.testing.assert_allclose(value, before['outputs'][key], atol=1e-10, rtol=1e-10)
        lines = ['# Local mixed-output Rust comparison', '',
                 f'{args.warmup} warmups; {args.samples} samples × {args.calls} calls; medians in ms/call.',
                 'Separate processes with the same workload and thread settings. Small differences may be measurement noise.', '',
                 '| Workload | Before ms | After ms | Speedup |', '| --- | ---: | ---: | ---: |']
        for key, timing in report['timings'].items():
            a, b = before['timings'][key]['median_ms'], timing['median_ms']
            lines.append(f'| {key} | {a:.6f} | {b:.6f} | {a/b:.2f}x |')
        lines += ['', '| Output | Max abs difference | Relative Frobenius error |', '| --- | ---: | ---: |']
        for key, value in report['accuracy_vs_before'].items():
            lines.append(f"| {key} | {value['max_abs']:.3g} | {value['relative_frobenius']:.3g} |")
        lines += ['', '| Rust vs NumPy | Max abs difference | Relative Frobenius error |', '| --- | ---: | ---: |']
        for key, value in report['rust_vs_numpy'].items():
            lines.append(f"| {key} | {value['max_abs']:.3g} | {value['relative_frobenius']:.3g} |")
        lines += ['', report['workload']['note'], '',
                  'Full model/seed/output selection, raw samples, numerical outputs, direct-vs-dense errors and extension hashes are in the JSON files.', '']
        args.output.with_suffix('.md').write_text('\n'.join(lines))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
