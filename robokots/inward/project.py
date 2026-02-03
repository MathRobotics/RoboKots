from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import json

from robokots.kots import Kots
from robokots.inward.expr import json_builder


@dataclass
class SolverConfig:
    type: str = "gauss_newton"
    max_iters: int = 20
    tol_r: float = 1e-10
    tol_dx: float = 1e-10


def load_project(path: str | Path) -> dict:
    project_path = Path(path)
    with project_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["_base_dir"] = str(project_path.parent)
    if "task" in data and "task_file" in data:
        raise ValueError("project: specify either 'task' or 'task_file', not both.")
    return data


def _resolve_model_path(model: dict, base_dir: str | None) -> str:
    if "path" not in model:
        raise ValueError("project.model.path is required")
    model_path = Path(model["path"])
    if not model_path.is_absolute() and base_dir is not None:
        model_path = Path(base_dir) / model_path
    return str(model_path)


def _solver_from_project(project: dict) -> SolverConfig:
    solver = project.get("solver", {})
    return SolverConfig(
        type=str(solver.get("type", "gauss_newton")),
        max_iters=int(solver.get("max_iters", 20)),
        tol_r=float(solver.get("tol_r", 1e-10)),
        tol_dx=float(solver.get("tol_dx", 1e-10)),
    )


def _variables_from_task(task: dict) -> list[dict]:
    variables = task.get("variables", None)
    if variables is None:
        raise ValueError("task.variables is required")

    if isinstance(variables, dict):
        return [{"name": name, "init": value} for name, value in variables.items()]

    if isinstance(variables, list):
        return list(variables)

    raise ValueError("task.variables must be a dict or list")


def _default_var_name(variables: list[dict], target: dict) -> str:
    if "var" in target:
        return str(target["var"])
    if len(variables) == 1:
        return str(variables[0]["name"])
    raise ValueError("target.var is required when multiple variables exist")


def _target_to_term(target: dict, *, default_var: str) -> dict:
    typ = str(target.get("type", "")).lower()
    if typ not in {"link_pos", "link_position"}:
        raise ValueError(f"unknown target type: {target.get('type')}")

    link = target.get("link", None)
    if link is None:
        raise ValueError("link_pos target requires 'link'")

    value = target.get("value", target.get("target", None))
    if value is None:
        raise ValueError("link_pos target requires 'value' (or legacy 'target')")

    name = target.get("name", f"{link}_pos_error")
    weight = float(target.get("weight", 1.0))

    expr = {
        "type": "sub",
        "name": name,
        "a": {
            "type": "get_state",
            "key": {
                "owner_type": "link",
                "owner_name": link,
                "dtype": "frame",
                "field": "pos",
            },
            "jac": {
                "var": default_var,
                "field": "pos_J_q",
            },
        },
        "b": {
            "type": "constant",
            "value": value,
            "vars": [default_var],
        },
    }

    if weight == 1.0:
        cost = {"type": "l2"}
    else:
        cost = {"type": "scalar", "w": weight}

    return {"expr": expr, "cost": cost}


def project_to_spec(project: dict) -> dict:
    task = project.get("task", None)
    task_file = project.get("task_file", None)
    if task is not None and task_file is not None:
        raise ValueError("Specify either 'task' or 'task_file', not both.")
    if task is None and task_file is None:
        raise ValueError("project.task or project.task_file is required")

    if task is None:
        base_dir = project.get("_base_dir", None)
        task_path = Path(task_file)
        if not task_path.is_absolute() and base_dir is not None:
            task_path = Path(base_dir) / task_path
        with task_path.open("r", encoding="utf-8") as f:
            task = json.load(f)

    variables = _variables_from_task(task)
    terms: list[dict] = []

    if "targets" in task:
        for tgt in task.get("targets", []):
            default_var = _default_var_name(variables, tgt)
            terms.append(_target_to_term(tgt, default_var=default_var))
    elif "terms" in task:
        terms = list(task.get("terms", []))
    else:
        raise ValueError("task must define 'targets' or 'terms'")

    spec: dict[str, Any] = dict(task)
    spec["variables"] = variables
    spec["terms"] = terms

    if "time" in project:
        spec["time"] = project["time"]

    return spec


def build_problem_from_project(project: dict, *, base_dir: str | None = None):
    model = project.get("model", {})
    model_path = _resolve_model_path(model, base_dir or project.get("_base_dir"))
    order = int(model.get("order", 1))

    kots = Kots.from_json_file(model_path, order=order)
    trajectory = project.get("trajectory", None)
    backend = SimpleNamespace(build_state=json_builder.make_build_state_kots(kots, trajectory=trajectory))

    spec = project_to_spec(project)
    problem, ctx = json_builder.build_problem_from_spec(spec, backend=backend)
    solver = _solver_from_project(project)

    return problem, ctx, solver


def build_problem_from_project_file(path: str | Path):
    project = load_project(path)
    base_dir = project.get("_base_dir")
    return build_problem_from_project(project, base_dir=base_dir)


__all__ = [
    "SolverConfig",
    "load_project",
    "project_to_spec",
    "build_problem_from_project",
    "build_problem_from_project_file",
]
