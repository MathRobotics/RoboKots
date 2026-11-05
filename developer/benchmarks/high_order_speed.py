"""Isolated high-order dynamics prototypes; no production extension installation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from .state_cache_scope import ROOT, RESULTS, load_probe
SUPPORT = Path(__file__).with_name('high_order_probe')
VARIANTS = ('current', 'spatial', 'transport', 'both')


def function_text(source, name):
    start = source.index(f'pub(crate) fn {name}(')
    brace = source.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    crate = directory / 'crate'
    crate.mkdir(exist_ok=True)
    original = ROOT / 'robokots/_rust'
    shutil.copytree(original/'src', crate/'src', dirs_exist_ok=True, ignore=shutil.ignore_patterns('._*'))
    for name in ('Cargo.toml', 'Cargo.lock'): shutil.copy2(original/name, crate/name)
    generic = crate/'src/cmtm_generic.rs'
    # Reconstruct the original dispatch in the isolated copy now that the
    # spatial recurrence is used by production. Keep the experiment comparable.
    generic_source = generic.read_text().replace(
        '        if order >= 4 {\n'
        '            self.kinematics_cmtm_high_order_into(motion, order, ws);\n'
        '            return;\n'
        '        }\n', '')
    generic.write_text(generic_source+'\n'+(SUPPORT/'kinematics.rs').read_text())
    spatial = crate/'src/spatial.rs'
    source = spatial.read_text()
    original_transport = function_text(source, 'cmtm_accumulate_mat_adj_wrench_series_into')
    signature = original_transport[:original_transport.index('{')]
    signature = signature.replace('fn cmtm_accumulate_mat_adj_wrench_series_into(', 'fn probe_accumulate_prepared_wrench_into(')
    for name in ('elem_mat', 'vecs', 'scaled_vecs'):
        signature = signature.replace('    '+name+':', '    _'+name+':')
    if 'pub(crate) fn cmtm_accumulate_wrench_series_from_blocks_into(' in source:
        body = ('    cmtm_accumulate_wrench_series_from_blocks_into(\n'
                '        raw_rhs, order, fact, a_blocks, c_blocks, raw_target,\n'
                '    );\n}')
    else:
        body = original_transport[original_transport.index('    for k in 0..order {'):]
    spatial.write_text(source+'\n'+signature+'{\n'+body+'\n')
    series = crate/'src/cmtm_series.rs'
    source = series.read_text()
    original_dynamics = function_text(source, 'dynamics_cmtm_into')
    if 'cmtm_accumulate_wrench_series_from_blocks_into(' in original_dynamics:
        # Restore the pre-optimization gravity transport in the baseline only.
        start = original_dynamics.index('                    // Momentum prepared')
        end = original_dynamics.index('                    );', start) + len('                    );')
        full_transport = '''                    cmtm_accumulate_mat_adj_wrench_series_into(
                        rel_mat,
                        &rel_vecs[..dynamics_order.saturating_sub(1) * 6],
                        child_gravity,
                        dynamics_order,
                        &ws.factorial,
                        &mut ws.tmp_scaled_vecs,
                        &mut ws.tmp_wrench_adj_a_blocks,
                        &mut ws.tmp_wrench_adj_c_blocks,
                        &mut ws.tmp_gravity_force,
                    );'''
        restored = original_dynamics[:start] + full_transport + original_dynamics[end:]
        source = source.replace(original_dynamics, restored, 1)
        original_dynamics = restored
    for variant in VARIANTS[1:]:
        body = original_dynamics.replace('fn dynamics_cmtm_into(', f'fn dynamics_probe_{variant}_into(', 1)
        if variant in ('transport', 'both'):
            # Same location despite method-name length change: locate again.
            start=body.index('let child_gravity =')
            pos=body.index('cmtm_accumulate_mat_adj_wrench_series_into(', start)
            body=body[:pos]+body[pos:].replace('cmtm_accumulate_mat_adj_wrench_series_into(', 'probe_accumulate_prepared_wrench_into(',1)
        if variant in ('spatial', 'both'):
            body=body.replace('self.kinematics_cmtm_into(motion, kin_order, &mut ws.cmtm);', 'self.probe_spatial_kinematics_into(motion, kin_order, &mut ws.cmtm);')
        source += '\nimpl RustCompiledRobot {\n'+body+'\n}\n'
    series.write_text(source)
    api=crate/'src/py_api.rs'; source=api.read_text()
    anchor='    /// Selected local/world momentum, force and torque products.'
    api.write_text(source.replace(anchor, (SUPPORT/'methods.rs').read_text()+'\n'+anchor, 1))
    env=dict(os.environ, PYO3_PYTHON=sys.executable, CARGO_TARGET_DIR=str(directory/'target'))
    cmd=['cargo','rustc','--offline','--release','--manifest-path',str(crate/'Cargo.toml')]
    if sys.platform=='darwin': cmd+=['--','-C','link-arg=-undefined','-C','link-arg=dynamic_lookup']
    subprocess.run(cmd,env=env,check=True)
    lib=directory/'target/release'/('librobokots_rust.dylib' if sys.platform=='darwin' else 'librobokots_rust.so')
    shutil.copy2(lib,directory/'probe.so')
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in original.glob('src/*.rs') if not p.name.startswith('._')}
    (directory/'source_hashes.json').write_text(json.dumps(hashes,indent=2))


def independent_checks(np, Kots, StateType):
    """Compare every link/joint momentum/force series with NumPy on an oblique,
    branched model containing fixed joints; do not use the same Rust formula."""
    rng=np.random.default_rng(397)
    records=[]
    for order in (4,5,6,8):
        k=Kots.from_urdf_file(str(ROOT/'tests/test_model/branched_fixed.urdf'),order=order)
        robot=k._rust_compiled_robot()
        for zero in (False,True):
            motion=np.zeros(k.dof()*order) if zero else rng.normal(scale=.2,size=k.dof()*order)
            gravity=np.array([.3,-.4,-9.81])
            k.import_motions(motion); k.dynamics(backend='numpy',gravity=gravity)
            _,actual=robot.benchmark_high_order(motion[None,:],order-2,gravity,'both',1)
            actual=np.asarray(actual)
            links=k.link_name_list(); joints=k.joint_name_list()
            offset=len(links)*16+len(links)*(order-1)*6+len(joints)*16+len(joints)*(order-1)*6
            expected=[]
            for owner,names,family,count in [('link',links,'momentum',order-1),('link',links,'force',order-2),('joint',joints,'momentum',order-1),('joint',joints,'force',order-2)]:
                for name in names:
                    for n in range(count):
                        key=family if n==0 else f'{family}_diff{n}'
                        expected.extend(np.asarray(k.state_info(StateType(owner,name,key))).reshape(-1))
            expected=np.asarray(expected); selected=actual[offset:offset+expected.size]
            np.testing.assert_allclose(selected,expected,atol=2e-9,rtol=2e-9)
            records.append(dict(order=order,zero_motion=zero,max_abs=float(np.max(np.abs(selected-expected))),relative_fro=float(np.linalg.norm(selected-expected)/max(np.linalg.norm(expected),1e-30))))
    return records


def run(args):
    load_probe(args.extension)
    import numpy as np
    from robokots.kots import Kots,StateType
    from robokots.core.robot import RobotStruct
    from .common import build_model
    rng=np.random.default_rng(917)
    checks=independent_checks(np,Kots,StateType)
    rows=[]
    for dof in (16,64):
        for batch in (1,8):
            for order in (4,5,6,8):
                k=Kots(RobotStruct.from_dict(build_model(dof,'humanoid')),order=order,dim=3)
                robot=k._rust_compiled_robot()
                motion=rng.normal(scale=.2,size=(batch,dof*order))
                for nonzero in (False,True):
                    gravity=np.array([.3,-.4,-9.81] if nonzero else [0,0,0],dtype=float)
                    def call(v,x=motion,loops=args.loops):
                        return robot.benchmark_high_order(x,order-2,gravity,v,loops)
                    first={v:call(v,loops=1) for v in VARIANTS}
                    errors={}
                    for v in VARIANTS:
                        actual=np.asarray(first[v][1]);ref=np.asarray(first['current'][1])
                        np.testing.assert_allclose(actual,ref,atol=5e-8,rtol=5e-9)
                        errors[v]=dict(max_abs=float(np.max(np.abs(actual-ref))),relative_fro=float(np.linalg.norm(actual-ref)/max(np.linalg.norm(ref),1e-30)))
                    # Check each sample separately and alternating inputs so stale
                    # scratch/zero-motion fields cannot pass a last-sample check.
                    for x in [*motion, np.zeros(dof*order), motion[0]*.71]:
                        ref=call('current',x[None,:],1)[1]
                        for v in VARIANTS[1:]:
                            np.testing.assert_allclose(call(v,x[None,:],1)[1],ref,atol=5e-8,rtol=5e-9)
                    for _ in range(args.warmup):
                        for v in rng.permutation(VARIANTS):call(str(v))
                    samples={v:[] for v in VARIANTS}
                    for _ in range(args.repeats):
                        for v in rng.permutation(VARIANTS):samples[str(v)].append((np.asarray(call(str(v))[0])*1e6).tolist())
                    variants={}
                    for v in VARIANTS:
                        data=np.asarray(samples[v])
                        variants[v]=dict(median_dynamics_us=float(np.median(data[:,0])),median_kinematics_us=float(np.median(data[:,1])),p25_dynamics_us=float(np.percentile(data[:,0],25)),p75_dynamics_us=float(np.percentile(data[:,0],75)),samples_us=data.tolist(),first_us=(np.asarray(first[v][0])*1e6).tolist(),error=errors[v])
                    rows.append(dict(dof=dof,batch=batch,order=order,gravity=gravity.tolist(),variants=variants))
                    print(f"DOF={dof} batch={batch} order={order} gravity={nonzero}: {variants['current']['median_dynamics_us']:.2f} -> {variants['both']['median_dynamics_us']:.2f} us",flush=True)
    data=dict(metadata=dict(measured_at=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),python=sys.version,numpy=np.__version__,rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),seed=917,warmup=args.warmup,repeats=args.repeats,loops_per_measurement=args.loops,dtype='float64',units='microseconds per batch',source_hashes=json.loads(args.extension.with_name('source_hashes.json').read_text())),independent_numpy_checks=checks,rows=rows)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS/'high_order_speed.json').write_text(json.dumps(data,indent=2)+'\n')
    report(data)


def report(data):
    m=data['metadata']
    lines=['# High-order dynamics speed experiment','', 'This is an isolated comparison against the reconstructed pre-optimization baseline. Production now uses the combined optimization; see [the production API comparison](high_order_production.md).','',f"Environment: {m['platform']}; Python {sys.version.split()[0]}, NumPy {m['numpy']}, {m['rustc']}. float64, seed {m['seed']}.",f"Warmup {m['warmup']}, randomized interleaved samples {m['repeats']}, {m['loops_per_measurement']} evaluations per timing. Median µs per batch. First-call kernel timings (one evaluation), quartiles and raw timings in JSON.",'',
           'Isolated prototype extension; production sources and installed extension unchanged. Timing excludes Python conversion, allocation of the persistent workspace, model construction and output extraction. Includes a temporary 3x3 rotation-series buffer in the spatial variant. The same complete state is computed in every dynamics variant; no outputs or derivative orders are omitted. Kinematics is timed separately, not subtracted from dynamics as if it were an exact profile.', '',
           'Variants: current = CMTM matrix-series composition/recovery; spatial = propagate ordinary body-velocity derivatives using the relative inverse-rotation series (fixed/revolute only); transport = reuse momentum transport blocks for gravity instead of rebuilding the same prefix; both = combine these changes.', '',
           '|DOF|Batch|Order|Gravity|Current dynamics|Spatial|Transport|Both|Speedup|Current kinematics|Spatial kinematics|',
           '|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in data['rows']:
        v=r['variants'];c=v['current'];b=v['both'];s=v['spatial'];t=v['transport']
        lines.append(f"|{r['dof']}|{r['batch']}|{r['order']}|{'nonzero' if any(r['gravity']) else 'zero'}|{c['median_dynamics_us']:.2f}|{s['median_dynamics_us']:.2f}|{t['median_dynamics_us']:.2f}|{b['median_dynamics_us']:.2f}|{c['median_dynamics_us']/b['median_dynamics_us']:.2f}x|{c['median_kinematics_us']:.2f}|{s['median_kinematics_us']:.2f}|")
    err=max(v['error']['max_abs'] for r in data['rows'] for v in r['variants'].values())
    rel=max(v['error']['relative_fro'] for r in data['rows'] for v in r['variants'].values())
    checks=data['independent_numpy_checks']
    lines+=['','## Correctness and limits','',f"Against existing Rust state: max absolute difference {err:.6g}, max relative Frobenius error {rel:.6g}. Compared all semantic buffers including poses, velocity series, link/joint momentum and force, torque and gravity intermediates. Each batch sample, zero motion and changed motion were checked independently.",f"Independent NumPy comparison on oblique branched/fixed-joint model, orders 4/5/6/8, random and zero motion: max absolute difference {max(x['max_abs'] for x in checks):.6g}, max relative Frobenius error {max(x['relative_fro'] for x in checks):.6g} for all link/joint momentum/force series.",
           'The new recurrence is analytically equivalent for fixed/revolute joints, and retains factorial scaling at series boundaries. It does not extend the current Rust model support to prismatic/spherical/floating joints. The prototype does not replace JVP/VJP kernels and their runtime was not measured here. Public Python API speedups can be smaller because conversion/dispatch cost is excluded. No derivative-result caching, dense Jacobian caching, parallel execution or additional per-link persistent cache is introduced.','']
    lines += ['## 検討結果と推奨順序', '',
        '第一候補は order 4 以降の運動学計算の置換。現行はCMTMの4×4行列系列を合成し、逆変換して速度系列へ戻す。試作は相対回転の逆行列系列を使い、親の空間速度系列から子の系列を直接計算する。次数に対する計算量のオーダーは同じでも、行列サイズと中間演算が減る。',
        '相対並進 p が関節座標に依存しない固定・回転関節に対して、ω_child = R_rel^T ω_parent + axis qdot、v_child = R_rel^T (v_parent − p × ω_parent) を高次の積の微分で評価する。回転系列は階乗正規化し、公開の速度系列では通常の時間微分へ戻す。',
        '第二候補は重力ありの場合の変換系列の共有。子の関節運動量の変換で構築したブロックの先頭部分を、同じ関節の重力系列の変換にも使う。新しい永続キャッシュではなく、一回の計算内の既存scratchの再利用。重力ゼロの場合にはこの重複自体がない。',
        '測定した組み合わせ案は約1.52～1.61倍の速度（約34～38%の時間短縮）。order 3 の専用高速経路には手を入れない。追加scratchは1サンプルの関節巡回で共用する3×3行列の系列のみで、order 4で216 bytes、order 8で504 bytes。リンク数やバッチ数に比例する永続状態を増やしていない。',
        '本体へ適用する場合は、最初に運動学の置換、次に重力変換共有を独立変更として検証する。固定関節・ゼロ姿勢・斜め軸・分岐・各次数・重力・バッチの状態値に加え、既存JVP/VJPカーネルとの整合性を確認する。今回の速度測定は状態計算のみで、微分APIの高速化倍率は未測定。',
        '前回の必要時補完キャッシュ案とは併用可能だが、削減する仕事に重なりがあるため、速度改善倍率を単純に掛け合わせることはできない。公開APIに組み込んだ状態で dynamics 単体と dynamics＋微分を再測定して判断する。',
        'このスクリプトは検討用の隔離拡張を作成し、通常利用するインストール済み拡張を変更しない。本体へ適用済みの結果は high_order_production.md を参照。', '']
    (RESULTS/'high_order_speed.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--build',type=Path)
    parser.add_argument('--extension',type=Path)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--loops',type=int,default=5)
    args=parser.parse_args()
    if args.build:build(args.build.resolve())
    elif args.extension:run(args)
    else:parser.error('choose --build or --extension')
