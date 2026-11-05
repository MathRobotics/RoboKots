"""Selected world/pose outputs: NumPy vs Rust with optional matched baseline.

--compare checks numerical equality with a corrected same-workload baseline.
Pre-correction world/pose results are not equivalent and must not be used for
speedup claims. --kinematics-only excludes dynamics and gravity.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
import robokots._rust_core as extension
from robokots.kots import Kots, StateType
from .core_state_layout import measure
from .kernel_layout import error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare',type=Path)
    parser.add_argument('--kinematics-only',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    model=root/'tests/test_model/branched_fixed.urdf'
    order=4
    gravity=[0.,0.,0.] if args.kinematics_only else [.2,-.3,-9.81]
    build_label="import_and_kinematics" if args.kinematics_only else "import_and_dynamics"
    states=[StateType('joint','a_elbow','acc','world'),
            StateType('link','a_tip','jerk','world'),
            StateType('link','b_payload','frame','world'),
            StateType('joint','a_shoulder','pos','local'),
            StateType('link','a_tip','force_diff1'),
            StateType('joint','b_shoulder','torque_diff1')]
    if args.kinematics_only:
        states=[s for s in states if not s.is_dynamics]
    timings={};outputs={};checks={};caches={}
    for backend in ('numpy','rust'):
        for shape in ((),(2,)):
            rng=np.random.default_rng(812)
            k=Kots.from_urdf_file(str(model),order=order)
            x=rng.normal(scale=.2,size=shape+(k.dof()*order,))
            name=f'{backend}/'+('batch2' if shape else 'single')
            def build():
                k.import_motions(x)
                if args.kinematics_only:
                    k.kinematics(backend=backend)
                else:
                    k.dynamics(backend=backend,gravity=gravity)
            timings[name+'/'+build_label]=measure(build,5,30,5)
            jac=k.jacobian(states)
            v=rng.normal(size=shape+(jac.shape[-1],));w=rng.normal(size=shape+(jac.shape[-2],))
            operations={'dense':lambda:k.jacobian(states),
                        'jvp':lambda:k.jacobian_mul(states,v),
                        'vjp':lambda:k.jacobian_transpose_mul(states,w)}
            with patch.object(k,'_jacobian_from_state',side_effect=AssertionError('dense fallback')):
                operations['jvp']();operations['vjp']()
            for op,fn in operations.items():
                timings[name+'/'+op]=measure(fn,5,30,5)
                outputs[name+'/'+op]=np.asarray(fn()).tolist()
            if backend=='rust':
                caches[name]=k._rust_selected_workspace_[2].cache_info()
            numerical=k.jacobian(states,numerical=True)
            checks[name+'/numerical']=error(jac,numerical)
            np.testing.assert_allclose(jac,numerical,atol=2e-6,rtol=2e-6)
            for op,expected in [('jvp',(jac@v[...,None])[...,0]),('vjp',(np.swapaxes(jac,-1,-2)@w[...,None])[...,0])]:
                checks[name+'/'+op]=error(outputs[name+'/'+op],expected)
                np.testing.assert_allclose(outputs[name+'/'+op],expected,atol=1e-10,rtol=1e-10)
    for key,value in outputs.items():
        if key.startswith('rust/'):
            checks[key+'/numpy']=error(value,outputs[key.replace('rust/','numpy/')])
            np.testing.assert_allclose(value,outputs[key.replace('rust/','numpy/')],atol=1e-10,rtol=1e-10)
    report=dict(environment=dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,mathrobo=version("mathrobo"),
                git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
                threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
                extension_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()),
                workload=dict(model=str(model.relative_to(root)),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                seed=812,order=order,dtype='float64',gravity=gravity,states=[vars(s) for s in states],
                warmup=5,samples=30,calls=5,kinematics_only=args.kinematics_only,note=__doc__,numerical='central difference, step 1e-8 (mathrobo numerical_grad)'),
                timings=timings,outputs=outputs,accuracy=checks,cache_info=caches)
    lines=['# Selected spatial outputs','',
           'Kinematics only.' if args.kinematics_only else 'Mixed kinematics/dynamics.',
           '5 warmups; 30 samples × 5 calls; medians in ms. Derivatives start from computed states; internal recomputation and Python boundary conversion are included.',
           'Derivative primals are cached across repeated calls. State construction is separate. Same model and float64 inputs; mixed dynamics uses nonzero gravity, kinematics-only uses none. No JAX/JIT measurements: the mixed pose contract is outside the current dynamics AD API.', '',
           '| Input | Operation | NumPy ms | Rust ms | NumPy / Rust |','| --- | --- | ---: | ---: | ---: |']
    for shape in ('single','batch2'):
        for op in (build_label,'dense','jvp','vjp'):
            a=timings[f'numpy/{shape}/{op}']['median_ms'];b=timings[f'rust/{shape}/{op}']['median_ms']
            lines.append(f'| {shape} | {op} | {a:.6f} | {b:.6f} | {a/b:.2f}x |')
    lines+=['','| Accuracy reference | Max abs | Relative Frobenius |','| --- | ---: | ---: |']
    for key,e in checks.items():lines.append(f"| {key} | {e['max_abs']:.3g} | {e['relative_frobenius']:.3g} |")
    lines+=['',__doc__,'','See the JSON for first-call timing, raw samples, outputs and environment.']
    if args.compare:
        before=json.loads(args.compare.read_text())
        for key in ('model_sha256','seed','order','dtype','gravity','states','warmup','samples','calls'):
            assert before['workload'][key]==report['workload'][key], key
        assert before['workload'].get('kinematics_only',False)==args.kinematics_only
        report['accuracy_vs_before']={key:error(value,before['outputs'][key]) for key,value in outputs.items()}
        for key,value in outputs.items():
            np.testing.assert_allclose(value,before['outputs'][key],atol=1e-10,rtol=1e-10)
        lines+=['','## Same-workload before/after','',
                '| Workload | Before ms | After ms | Speedup |','| --- | ---: | ---: | ---: |']
        for key,value in timings.items():
            a=before['timings'][key]['median_ms'];b=value['median_ms']
            lines.append(f'| {key} | {a:.6f} | {b:.6f} | {a/b:.2f}x |')
    lines+=['','Cache counters are (kinematics primal evaluations, dynamics primal evaluations, cached samples):',str(caches)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(args.output)


if __name__=='__main__':main()
