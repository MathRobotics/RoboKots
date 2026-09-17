"""Compare preserved FK-time-AD measurements with a new ordinary-ID-time-AD run.

No measurements are rerun or overwritten. Inputs/environments must match.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .dynamics_autodiff_compare import error


def compare(fk, id_run):
    if id_run["metadata"]["rows"] != [7]:
        raise ValueError("This comparison requires the single joint-torque selection (rows=[7])")
    for key in ("platform", "dof", "seed", "gravity", "model", "motions", "rows",
                "derivatives", "repeats", "warmup", "check_values"):
        if fk["metadata"].get(key) != id_run["metadata"].get(key):
            raise ValueError(f"Incompatible measurement condition: {key}")
    if not fk["results"] or not id_run["results"]:
        raise ValueError("Both runs need completed results")
    environment = fk["results"][0]["environment"]
    if any(r["environment"] != environment for run in (fk, id_run) for r in run["results"]):
        raise ValueError("Incompatible measurement environment")
    rows = []
    direct_errors = []
    for k in id_run["metadata"]["derivatives"]:
        for direction in ("forward", "reverse"):
            old_result = new_result = None
            for label, prefix, run in (
                ("Derivative coefficients + AD (new run)", "", id_run),
                ("FK + time AD (previous run)", "time_", fk),
                ("Ordinary ID + time AD (new run)", "id_time_", id_run),
            ):
                method = f"{prefix}{direction}_jit"
                matches = [r for r in run["results"] if r["k"] == k and r["method"] == method and r["row"] == 7]
                result = matches[0] if matches else None
                failures = [f for f in run["failures"] if f["k"] == k and f["method"] == method and f["row"] == 7]
                rows.append({"k": k, "direction": direction, "formulation": label,
                             "result": result, "failure": failures[0] if failures else None})
                if prefix == "time_":
                    old_result = result
                elif prefix == "id_time_":
                    new_result = result
            if old_result is not None and new_result is not None:
                direct_errors.append({"k": k, "direction": direction, "errors": {
                    key: error(np.asarray(new_result[key]), np.asarray(old_result[key]))
                    for key in ("value", "other_value", "jacobian", "other_jacobian")}})
    return {"fk_utc": fk["metadata"]["utc"], "id_utc": id_run["metadata"]["utc"],
            "dof": id_run["metadata"]["dof"], "repeats": id_run["metadata"]["repeats"],
            "warmup": id_run["metadata"]["warmup"],
            "rows": rows, "direct_errors_id_vs_fk": direct_errors}


def render(comparison):
    lines = ["# Time AD starting from FK vs ordinary ID", "",
             f"Preserved FK run: {comparison['fk_utc']}. New ID/coefficient run: {comparison['id_utc']}.", "",
             "The previous FK implementation and results are preserved, not rerun. "
             "Model, motions, gravity, seed, dtype/software/device environment, warmups and repeat counts "
             "were checked for equality. These are separate local runs, not simultaneous measurements.", "",
             f"All cases use the first actuated joint torque of the same {comparison['dof']}-DOF rigid serial arm. "
             f"k denotes ordinary torque time-derivative order. At k=4 the outer Jacobian is 1x{7 * comparison['dof']}. "
             "All methods use dense outer AD. For the time-AD formulations, inner time AD is "
             "forward JVP in both outer modes.", "",
             "Derivative coefficients + AD computes analytic high-order time derivatives directly on coefficient series before outer AD. "
             "This is an implementation representation of the same high-order algebra as explicit CMTM matrices, "
             "not a separate mathematical differentiation method. Both representations use coefficient recurrences. "
             "FK + time AD obtains velocity/momentum rate via AD from FK, then differentiates torque in time. "
             "Ordinary ID + time AD always evaluates ID(q,qdot,qddot) at motion order 3, "
             "then applies k total-time JVPs before outer AD. It does not reuse higher-order torque recurrences.", "",
             f"First time includes JIT tracing, compilation and execution. Warm time is the median of {comparison['repeats']} "
             f"calls after the first call and {comparison['warmup']} warmups, including input conversion, synchronization and "
             "NumPy output. Peak RSS includes compilation and runtime caches, excludes post-measurement "
             "validation, and is not per-call memory. Missing results are never substituted by another method.", "",
             "| k | Outer AD | Formulation | First s | Warm ms | Peak MiB |",
             "|---:|---|---|---:|---:|---:|"]
    for row in comparison["rows"]:
        r = row["result"]
        head = f"| {row['k']} | {row['direction']} | {row['formulation']} |"
        if r is not None:
            lines.append(head + f" {r['first_ms']/1000:.3f} | {r['median_ms']:.3f} | {r['peak_mib']:.1f} |")
        else:
            reason = row["failure"]["error"] if row["failure"] else "not measured"
            lines.append(head + f" {'timeout' if 'timed out' in reason else 'not completed'} | — | — |")
    lines += ["", "## Agreement", "",
              "Each run checks two inputs against NumPy analytic values and Jacobians (see source reports). "
              "The following directly compares the new ID formulation with the preserved FK formulation. "
              "Relative Frobenius errors and full failure details are in the JSON.", "",
              "| k | Outer AD | Max value difference | Max Jacobian difference |",
              "|---:|---|---:|---:|"]
    for item in comparison["direct_errors_id_vs_fk"]:
        e = item["errors"]
        lines.append(f"| {item['k']} | {item['direction']} | "
                     f"{max(e[k]['max_abs'] for k in ('value', 'other_value')):.3e} | "
                     f"{max(e[k]['max_abs'] for k in ('jacobian', 'other_jacobian')):.3e} |")
    return "\n".join(lines) + "\n"


def main():
    root = Path(__file__).with_name("results")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fk", type=Path, default=root / "time_autodiff_torque7.json")
    parser.add_argument("--id", type=Path, default=root / "id_time_autodiff_torque7.json")
    parser.add_argument("--output", type=Path, default=root / "time_ad_formulations")
    args = parser.parse_args()
    comparison = compare(json.loads(args.fk.read_text()), json.loads(args.id.read_text()))
    comparison["source_files"] = {"fk": str(args.fk), "id": str(args.id)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8")
    args.output.with_suffix(".md").write_text(render(comparison), encoding="utf-8")
    print(args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
