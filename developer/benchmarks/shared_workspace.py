"""Compare eager duplicate vs shared/lazy production workspace binaries."""
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

PHASES = ('create', 'first_dynamics', 'raw_kinematics', 'raw_dynamics', 'raw_pair', 'public_dynamics')


def worker(args):
    # Both libraries expose the same module name. Keep each module/class alive;
    # each Kots receives its own compiled robot, so no dispatch crosses binaries.
    modules = {}
    for name in ('baseline', 'optimized'):
        load_probe(getattr(args, name))
        modules[name] = sys.modules['robokots._rust_core']
    import numpy as np
    from robokots.kots import Kots, StateType
    from robokots.core.robot import RobotStruct
    from robokots.outward.rust.model import _model_data_from_robot
    from .common import build_model
    rng = np.random.default_rng(921)
    rows = {name: [] for name in modules}

    def timed(call, loops=1):
        t = perf_counter_ns()
        for _ in range(loops): call()
        return (perf_counter_ns()-t)/loops/1000

    def setup(module, dof, batch, order, motion, gravity):
        k = Kots(RobotStruct.from_dict(build_model(dof, 'humanoid')), order=order, dim=3)
        robot = module.RustCompiledRobot.from_model_data(_model_data_from_robot(k.robot_))
        k._rust_compiled_robot_ = robot
        def create(): return robot.create_batch_outward_data(order, batch)
        def bytes_of(data):
            method = getattr(data, '_workspace_buffer_bytes', None)
            return list(method()) if method else None
        data = create()
        memory = {'created': bytes_of(data)}
        data.compute_kinematics(motion)
        memory['kinematics'] = bytes_of(data)
        data.compute_dynamics(motion, gravity)
        memory['dynamics'] = bytes_of(data)
        data.compute_kinematics(motion)
        memory['kinematics_after_dynamics'] = bytes_of(data)
        def public():
            k.import_motions(motion)
            k.dynamics(backend='rust', gravity=gravity)
        def pair():
            data.compute_kinematics(motion)
            data.compute_dynamics(motion, gravity)
        calls = {'raw_kinematics': lambda: data.compute_kinematics(motion),
                 'raw_dynamics': lambda: data.compute_dynamics(motion, gravity),
                 'raw_pair': pair, 'public_dynamics': public}
        return k, create, memory, calls

    for dof in (16, 64):
        for batch in (1, 8):
            for order in (3, 5, 8):
                motion = rng.normal(scale=.2, size=(batch, dof*order))
                for nonzero in (False, True):
                    gravity = np.array([.3, -.4, -9.81] if nonzero else [0, 0, 0], dtype=float)
                    contexts = {name: setup(module, dof, batch, order, motion, gravity) for name,module in modules.items()}
                    first = {name: {phase: timed(call) for phase,call in ctx[3].items()} for name,ctx in contexts.items()}
                    for _ in range(args.warmup):
                        for name in rng.permutation(list(contexts)):
                            for call in contexts[name][3].values(): call()
                    samples = {name: {phase: [] for phase in PHASES} for name in contexts}
                    for _ in range(args.repeats):
                        for phase in rng.permutation(PHASES[2:]):
                            for name in rng.permutation(list(contexts)):
                                samples[name][phase].append(timed(contexts[name][3][phase], args.loops))
                        for name in rng.permutation(list(contexts)):
                            t = perf_counter_ns(); cold = contexts[name][1]()
                            samples[name]['create'].append((perf_counter_ns()-t)/1000)
                            cold.compute_kinematics(motion)
                            samples[name]['first_dynamics'].append(timed(lambda: cold.compute_dynamics(motion, gravity)))
                            del cold
                    for variant,(k,_,memory,_) in contexts.items():
                        values = []
                        for owner,names in [('link',k.link_name_list()),('joint',k.joint_name_list())]:
                            for name in names:
                                for family,count in [('momentum',order-1),('force',order-2)]:
                                    for n in range(count):
                                        key = family if n==0 else f'{family}_diff{n}'
                                        values.extend(np.asarray(k.state_info(StateType(owner,name,key))).reshape(-1))
                        rows[variant].append(dict(dof=dof, links=len(k.robot_.links), joints=len(k.robot_.joints), batch=batch, order=order, gravity=gravity.tolist(), memory_bytes=memory, first_us=first[variant], samples_us=samples[variant], values=values))
    args.output.write_text(json.dumps(rows))


def run(args):
    import numpy as np
    variants={'baseline':[], 'optimized':[]}
    with tempfile.TemporaryDirectory(prefix='shared-workspace-') as directory:
        for round_id in range(args.rounds):
            path=Path(directory)/f'round-{round_id}.json'
            subprocess.run([sys.executable,'-m','developer.benchmarks.shared_workspace','--baseline',str(args.baseline),'--optimized',str(args.optimized),'--output',str(path),'--warmup',str(args.warmup),'--repeats',str(args.repeats),'--loops',str(args.loops)],cwd=ROOT,check=True)
            result=json.loads(path.read_text())
            for name in variants: variants[name].append(result[name])
            print(f'Paired round {round_id+1}/{args.rounds}',flush=True)
    rows=[]
    for i,base in enumerate(variants['baseline'][0]):
        row={key:base[key] for key in ('dof','links','joints','batch','order','gravity')}
        row['variants']={}
        ref=np.asarray(base['values'])
        for name,rounds in variants.items():
            samples={p:[x for r in rounds for x in r[i]['samples_us'][p]] for p in PHASES}
            errors=[]
            for r in rounds:
                actual=np.asarray(r[i]['values'])
                np.testing.assert_allclose(actual,ref,atol=5e-8,rtol=5e-9)
                errors.append(dict(max_abs=float(np.max(np.abs(actual-ref))),relative_fro=float(np.linalg.norm(actual-ref)/max(np.linalg.norm(ref),1e-30))))
            row['variants'][name]=dict(samples_us=samples,median_us={p:float(np.median(x)) for p,x in samples.items()},first_us=[r[i]['first_us'] for r in rounds],memory_bytes=rounds[0][i]['memory_bytes'],errors=errors)
        rows.append(row)
    metadata=dict(measured_at=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),python=sys.version,numpy=np.__version__,rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),seed=921,rounds=args.rounds,warmup=args.warmup,repeats=args.repeats,loops=args.loops,dtype='float64',binary_sha256={name:hashlib.sha256(getattr(args,name).read_bytes()).hexdigest() for name in variants},source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'robokots/_rust/src').glob('*.rs') if not p.name.startswith('._')})
    baseline_metadata=args.baseline.with_name('baseline_metadata.json')
    if baseline_metadata.exists(): metadata['baseline_build']=json.loads(baseline_metadata.read_text())
    RESULTS.mkdir(exist_ok=True)
    (RESULTS/'shared_workspace.json').write_text(json.dumps(dict(metadata=metadata,rows=rows),indent=2)+'\n')
    lines=['# Shared kinematics and lazy dynamics workspace','',f"Environment: {metadata['platform']}, Python {sys.version.split()[0]}, NumPy {np.__version__}, {metadata['rustc']}. float64, seed 921.",f'{args.rounds} rounds in fresh processes; each process loads both binaries and randomly interleaves baseline/optimized per phase and sample. Per case: {args.warmup} warmups, {args.repeats} samples per round, {args.loops} calls per hot sample. Medians in microseconds per batch. Creation and first dynamics are one call per sample. Full samples, first hot-phase calls and hashes are in JSON.','', 'create: workspace allocation including model cloning; first dynamics: after kinematics on a fresh workspace, including lazy allocation. Hot raw calls include Python binding cost. Pair is kinematics + dynamics on the same workspace. Public dynamics includes import_motions and cache invalidation each time. Model compilation and output extraction are excluded. No derivative timings are claimed.','', '|DOF|Batch|Order|Gravity|Create before/after|First dynamics before/after|Hot kinematics before/after|Hot dynamics before/after|Pair before/after|Public dynamics before/after|','|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        b=r['variants']['baseline']['median_us'];a=r['variants']['optimized']['median_us']
        values='|'.join(f'{b[p]:.2f} / {a[p]:.2f}' for p in PHASES)
        lines.append(f"|{r['dof']}|{r['batch']}|{r['order']}|{'nonzero' if any(r['gravity']) else 'zero'}|{values}|")
    errors=[e for r in rows for e in r['variants']['optimized']['errors']]
    lines+=['',f"All local link/joint momentum and force derivative outputs: max absolute difference {max(e['max_abs'] for e in errors):.6g}, max relative Frobenius error {max(e['relative_fro'] for e in errors):.6g}.",'','## Numeric buffer capacity','', 'Diagnostic values count Vec capacity times element size; exclude object/allocator overhead, robot storage, Python outputs and temporary per-call allocations. Baseline binary has no diagnostic; its capacities below are from the previous recorded cache-scope experiment, not a new RSS measurement.','']
    r=next(r for r in rows if r['dof']==64 and r['batch']==8 and r['order']==5)
    m=r['variants']['optimized']['memory_bytes']
    lines+=['64 DOF, 66 links, 65 joints, batch 8, order 5:','', '|Stage|Baseline numeric buffers KiB|Optimized measured KiB|','|---|---:|---:|']
    for stage,values in m.items(): lines.append(f'|{stage}|1398.81|{sum(values)/1024:.2f}|')
    lines+=['','The shared CMTM allocation is retained through all transitions; dynamics-only arrays have zero capacity before first dynamics and remain reusable afterwards. Batch-shape cache limits and reduction of unused CMTM scratch arrays are outside this change.','']
    (RESULTS/'shared_workspace.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','optimized','output'): p.add_argument('--'+name,type=Path)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--loops',type=int,default=5)
    args=p.parse_args()
    if args.baseline and args.optimized and args.output: worker(args)
    elif args.baseline and args.optimized: run(args)
    else: p.error('provide baseline/optimized binaries; output selects a paired worker')
