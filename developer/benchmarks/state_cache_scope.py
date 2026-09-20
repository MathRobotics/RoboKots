"""Measure cache policies in an isolated Rust extension; never install it.

Build: .venv/bin/python -m developer.benchmarks.state_cache_scope --build /tmp/robokots-cache-probe
Run:   .venv/bin/python -m developer.benchmarks.state_cache_scope --extension /tmp/robokots-cache-probe/probe.so
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = Path(__file__).with_name("cache_probe")
RESULTS = Path(__file__).with_name("results")
POLICIES = ("current", "eager", "lazy_fill", "lazy_recompute")
PHASES = ("dynamics", "promotion", "first_jvp", "first_vjp", "further_jvp", "further_vjp")


def duplicate_function(path, name, replacement, edits):
    source = path.read_text()
    start = source.index(f"    pub(crate) fn {name}(")
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    original = source[start:end]
    new = original.replace(f"fn {name}(", f"fn {replacement}(", 1)
    for old, value in edits:
        assert old in new, (name, old)
        new = new.replace(old, value)
    path.write_text(source[:end] + "\n\n" + new + source[end:])


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    crate = directory / "crate"
    crate.mkdir(exist_ok=True)
    source = ROOT / "robokots/_rust"
    shutil.copytree(source / "src", crate / "src", dirs_exist_ok=True, ignore=shutil.ignore_patterns("._*"))
    for name in ("Cargo.toml", "Cargo.lock"):
        shutil.copy2(source / name, crate / name)
    generic = crate / "src/cmtm_generic.rs"
    series = crate / "src/cmtm_series.rs"
    outputs = crate / "src/dynamics_outputs.rs"
    duplicate_function(generic, "kinematics_cmtm_tangent_into", "kinematics_cmtm_tangent_prepared_into", [
        ("        self.kinematics_cmtm_into(motion, order, primal);\n", ""),
    ])
    duplicate_function(generic, "kinematics_cmtm_outward_reverse_into", "kinematics_cmtm_outward_reverse_prepared_into", [
        ("        self.kinematics_cmtm_into(motion, order, primal);\n", ""),
    ])
    duplicate_function(series, "dynamics_cmtm_link_tangent_into", "dynamics_cmtm_link_tangent_prepared_into", [
        ("        self.dynamics_cmtm_into(motion, dynamics_order, gravity, primal);\n", ""),
        ("self.kinematics_cmtm_tangent_into(", "self.kinematics_cmtm_tangent_prepared_into("),
    ])
    duplicate_function(series, "dynamics_cmtm_reverse_from_state_into", "dynamics_cmtm_reverse_prepared_into", [
        ("self.kinematics_cmtm_outward_reverse_into(", "self.kinematics_cmtm_outward_reverse_prepared_into("),
    ])
    duplicate_function(outputs, "dynamics_selected_tangent_into", "dynamics_selected_tangent_prepared_into", [
        ("self.dynamics_cmtm_link_tangent_into(", "self.dynamics_cmtm_link_tangent_prepared_into("),
    ])
    duplicate_function(outputs, "dynamics_selected_reverse_into", "dynamics_selected_reverse_prepared_into", [
        ("        self.dynamics_cmtm_into(motion, dynamics_order, gravity, primal);\n", ""),
        ("self.dynamics_cmtm_reverse_from_state_into(", "self.dynamics_cmtm_reverse_prepared_into("),
    ])
    api = crate / "src/py_api.rs"
    text = api.read_text()
    anchor = "    /// Selected local/world momentum, force and torque products."
    text = text.replace(anchor, (SUPPORT / "methods.rs").read_text() + "\n" + anchor, 1)
    text += "\n" + (SUPPORT / "helpers.rs").read_text()
    api.write_text(text)
    workspace = crate / "src/workspace.rs"
    text = workspace.read_text()
    memory_methods = "\nfn probe_vec_bytes<T>(v: &Vec<T>) -> usize { v.capacity() * std::mem::size_of::<T>() }\n"
    for name in ("CmtmWorkspace", "DynamicsCmtmWorkspace", "DynamicsCmtmTangentWorkspace"):
        body = text.split(f"struct {name} {{", 1)[1].split("\n}", 1)[0]
        fields = re.findall(r"pub\(crate\) (\w+): Vec<", body)
        terms = [f"probe_vec_bytes(&self.{field})" for field in fields]
        if name == "DynamicsCmtmWorkspace":
            terms += ["self.cmtm.probe_bytes()"]
        memory_methods += f"impl {name} {{ pub(crate) fn probe_bytes(&self) -> usize {{ {' + '.join(terms)} }} }}\n"
    fields = ("link_momentum", "link_force", "joint_momentum", "joint_force", "joint_gravity_force", "link_local_gravity", "joint_torque", "cached_motion", "factorial")
    terms = [f"probe_vec_bytes(&self.{field})" for field in fields]
    terms += [f"probe_vec_bytes(&self.cmtm.{field})" for field in ("link_mat", "link_vecs", "joint_mat", "joint_vecs")]
    memory_methods += f"impl DynamicsCmtmWorkspace {{ pub(crate) fn probe_semantic_bytes(&self) -> usize {{ {' + '.join(terms)} }} }}\n"
    workspace.write_text(text + memory_methods)
    env = dict(os.environ, PYO3_PYTHON=sys.executable, CARGO_TARGET_DIR=str(directory / "target"))
    command = ["cargo", "rustc", "--offline", "--release", "--manifest-path", str(crate / "Cargo.toml")]
    if sys.platform == "darwin":
        command += ["--", "-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]
    subprocess.run(command, env=env, check=True)
    library = directory / "target/release" / ("librobokots_rust.dylib" if sys.platform == "darwin" else "librobokots_rust.so")
    shutil.copy2(library, directory / "probe.so")
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("src/*.rs") if not p.name.startswith("._")}
    (directory / "source_hashes.json").write_text(json.dumps(hashes, indent=2))
    print(directory / "probe.so", flush=True)


def load_probe(path):
    spec = importlib.util.spec_from_file_location("robokots._rust_core", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def run(args):
    load_probe(args.extension)
    import numpy as np
    from robokots.kots import Kots
    from robokots.core.robot import RobotStruct
    from .common import build_model
    metadata = dict(measured_at=datetime.now(timezone.utc).isoformat(), rustc=subprocess.check_output(["rustc", "--version"], text=True).strip(), platform=platform.platform(), machine=platform.machine(), python=sys.version, numpy=np.__version__, seed=872,
                    warmup=args.warmup, repeats=args.repeats, reuse=args.reuse, rhs_cols=2,
                    dtype="float64", units="microseconds", timing="Rust kernels; excludes Python conversions, model compilation and initial persistent-buffer allocation; includes derivative workspace allocation",
                    first_call="one run per policy, after model compilation and before warmup",
                    source_hashes=json.loads(args.extension.with_name("source_hashes.json").read_text()))
    rows = []
    rng = np.random.default_rng(872)
    for dof in (16, 64):
        for batch in (1, 8):
            for order in (3, 5):
                for nonzero in (False, True):
                    gravity = np.array([0.3, -0.4, -9.81] if nonzero else [0, 0, 0], dtype=float)
                    kots = Kots(RobotStruct.from_dict(build_model(dof, "humanoid")), order=order, dim=3)
                    robot = kots._rust_compiled_robot()
                    motion = rng.normal(scale=.3, size=(batch, dof * order))
                    directions = rng.normal(size=(batch, dof * order, 2))
                    # Last link, a nonleaf joint, fixed joint and torque: mixed world/local.
                    active = [i for i, j in enumerate(kots.robot_.joints) if j.dof]
                    fixed = [i for i, j in enumerate(kots.robot_.joints) if not j.dof][0]
                    outputs = [(0, len(kots.robot_.links)-1, 1, order-3, True),
                               (1, active[0], 1, order-3, True),
                               (1, fixed, 0, order-2, False),
                               (1, active[-1], 2, order-3, False)]
                    weights = rng.normal(size=(batch, 19, 2))
                    def call(policy, x=motion):
                        return robot.benchmark_state_cache(x, directions, weights, outputs, order-2, gravity, policy, args.reuse)
                    first = {p: call(p) for p in POLICIES}
                    errors = {}
                    for p in POLICIES:
                        errors[p] = {}
                        for i, family in ((1, "jvp"), (2, "vjp")):
                            value = np.asarray(first[p][i]); reference = np.asarray(first["current"][i])
                            np.testing.assert_allclose(value, reference, atol=3e-8, rtol=3e-9)
                            errors[p][family] = dict(max_abs=float(np.max(np.abs(value-reference))), relative_fro=float(np.linalg.norm(value-reference) / max(np.linalg.norm(reference), 1e-30)))
                    # Distinct inputs and zero poses validate that skipping recomputation
                    # really uses the state produced for the current sample.
                    for x in (motion * .7, np.zeros_like(motion)):
                        reference = call("current", x)
                        for p in POLICIES[1:]:
                            actual = call(p, x)
                            for i in (1, 2):
                                np.testing.assert_allclose(actual[i], reference[i], atol=3e-8, rtol=3e-9)
                    for _ in range(args.warmup):
                        for p in rng.permutation(POLICIES): call(str(p))
                    samples = {p: [] for p in POLICIES}
                    for _ in range(args.repeats):
                        for p in rng.permutation(POLICIES):
                            samples[str(p)].append(np.asarray(call(str(p))[0]) * 1e6)
                    policies = {}
                    for p in POLICIES:
                        data = np.asarray(samples[p]); total = data.sum(axis=1)
                        policies[p] = dict(first_us=(np.asarray(first[p][0]) * 1e6).tolist(), samples_us=data.tolist(),
                                           median_us=dict(zip(PHASES, np.median(data, axis=0).tolist())),
                                           total_median_us=float(np.median(total)), total_p25_us=float(np.percentile(total,25)), total_p75_us=float(np.percentile(total,75)),
                                           error=errors[p])
                    row = dict(dof=dof, links=len(kots.robot_.links), joints=len(kots.robot_.joints), batch=batch, order=order, gravity=gravity.tolist(),
                               memory_bytes=dict(zip(("dynamics_buffers", "separate_kinematics_buffers", "semantic_state", "one_tangent_workspace"), first["current"][3])), policies=policies)
                    rows.append(row)
                    print(f"dof={dof} batch={batch} order={order} gravity={nonzero}: current={policies['current']['total_median_us']:.1f} lazy={policies['lazy_fill']['total_median_us']:.1f} us", flush=True)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "state_cache_scope.json"
    path.write_text(json.dumps(dict(metadata=metadata, rows=rows), indent=2) + "\n")
    report(metadata, rows)
    print(path, flush=True)


def report(metadata, rows):
    lines = ["# Rust state-cache scope experiment", "", "This is an isolated prototype; production dynamics and installed extension are unchanged.", "",
             f"Environment: {metadata['platform']}, Python {sys.version.split()[0]}, NumPy {metadata['numpy']}. Seed 872, float64, RHS columns 2.",
             f"Warmup {metadata['warmup']}, interleaved randomized measurements {metadata['repeats']}; {metadata['reuse']} JVP/VJP pairs after each dynamics calculation.",
             "All times are medians in microseconds. Rust kernel timers exclude Python conversion/dispatch, model compilation, and initial persistent-buffer allocation; derivative buffer allocation is included. First-call timings and raw samples are in the JSON.", "",
             "Policies: current = existing lean/full dynamics followed by independent derivative recomputation; eager = full state computed unconditionally; lazy_fill = keep existing dynamics and fill omitted fields on first derivative only; lazy_recompute = keep dynamics but recompute full state once on first derivative. Prepared derivatives skip repeated dynamics and kinematics evaluations.", "",
             "| DOF | Batch | Order | Gravity | Dynamics current | Dynamics eager | Dynamics lazy | Promotion fill | Promotion recompute | Total current | Total eager | Total lazy |", "|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        p=r['policies']; c=p['current']; e=p['eager']; l=p['lazy_fill']; lr=p['lazy_recompute']
        lines.append(f"|{r['dof']}|{r['batch']}|{r['order']}|{'nonzero' if any(r['gravity']) else 'zero'}|{c['median_us']['dynamics']:.2f}|{e['median_us']['dynamics']:.2f}|{l['median_us']['dynamics']:.2f}|{l['median_us']['promotion']:.2f}|{lr['median_us']['promotion']:.2f}|{c['total_median_us']:.2f}|{e['total_median_us']:.2f}|{l['total_median_us']:.2f}|")
    lines += ["", "## Memory", "", "Numeric Vec capacities only (KiB): excludes model copies, Vec/struct headers, allocator overhead, Python views and output arrays. Semantic state is a subset of dynamics buffers, not an additional copy. Tangent buffer is one sample with two RHS columns; it can be reused across batch samples. Production shares the kinematics allocation; the separate-kinematics column is zero. Scratch reduction is not measured here.", "", "|DOF|Batch|Order|Dynamics buffers|Separate kinematics|Semantic subset|One tangent buffer|", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if any(r['gravity']): continue
        m=r['memory_bytes']; lines.append(f"|{r['dof']}|{r['batch']}|{r['order']}|"+'|'.join(f'{v/1024:.2f}' for v in m.values())+'|')
    max_abs=max(p['error'][family]['max_abs'] for r in rows for p in r['policies'].values() for family in ('jvp','vjp'))
    rel=max(p['error'][family]['relative_fro'] for r in rows for p in r['policies'].values() for family in ('jvp','vjp'))
    lines += ["", "## Numerical agreement and limits", "", f"Against the existing selected Rust kernels: maximum absolute difference {max_abs:.3g}, maximum relative Frobenius error {rel:.3g}. Also checked distinct motion and all-zero motion for each case. This is a reference implementation, not an exact solution.",
              "The experiment isolates cache reuse; it does not measure JAX or numerical differentiation speed. Production regression tests independently compare selected derivatives with NumPy, central differences and JAX. No production cache invalidation or promotion policy has been changed by this experiment.", ""]
    lines += ["## 判断と保持範囲", "",
        "常時全状態を計算する eager 案は採用しない。order 3・重力ゼロでは現行 dynamics が省略している計算を復活させ、今回の測定では dynamics 単体が約1.39～1.60倍になった。",
        "推奨は lazy_fill。通常の dynamics は同じ計算経路を維持し、微分が最初に要求されたときだけ、既存のリンク運動・関節運動量から不足する関節情報、リンク運動量・力、関節力を補う。order 5や非ゼロ重力では元々計算済みの状態をそのまま利用する。",
        "補完先の数値領域は現在の DynamicsCmtmWorkspace に既に確保されており、今回の試作では状態保存用の追加配列を作っていない。保持するのは motion/重力/次数に依存する状態。JVP/VJPの右辺に依存する配列は計算用バッファとして分離し、密ヤコビ行列・world変換後の全系列を無条件に状態キャッシュへ追加しない。",
        "同一条件の新旧状態を重複保持せず、motion更新時は既存領域を更新する。モデル・motion revision・次数・重力・batch shape を有効性条件とし、異なる形状を無制限には蓄積しない。最新の利用中状態と必要な作業バッファを優先する。キャッシュ上限や形状変更時の割り当て性能は今回の測定対象外。",
        "運動学領域の一本化と動力学固有領域の遅延確保は本体へ適用済み。別確保の運動学領域はゼロ。領域の確保・切り替え・定常更新の測定は shared_workspace.md を参照。この実験は dynamics 後の微分で状態を再利用する方針を比較する。", "",
        "### dynamics + 最初のJVP/VJP各1回", "",
        "以下も同じ測定の先頭4フェーズの和の中央値。単位はµs。", "",
        "|DOF|Batch|Order|Gravity|Current|Lazy fill|", "|---:|---:|---:|:---:|---:|---:|"]
    import numpy as np
    for r in rows:
        c = np.median(np.sum(np.asarray(r['policies']['current']['samples_us'])[:, :4], axis=1))
        l = np.median(np.sum(np.asarray(r['policies']['lazy_fill']['samples_us'])[:, :4], axis=1))
        lines.append(f"|{r['dof']}|{r['batch']}|{r['order']}|{'nonzero' if any(r['gravity']) else 'zero'}|{c:.2f}|{l:.2f}|")
    lines += ["", "本体への適用は未実施。この結果はキャッシュ範囲の判断用の試作測定であり、公開API全体の速度向上を保証する値ではない。", ""]
    (RESULTS / 'state_cache_scope.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path)
    parser.add_argument('--extension', type=Path)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--reuse', type=int, default=4)
    args = parser.parse_args()
    if args.build: build(args.build.resolve())
    elif args.extension: run(args)
    else: parser.error('choose --build or --extension')
