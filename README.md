# RoboKots

Utilities based on the Kots framework for robotic system modeling, motion planning and optimization.

Clone RoboKots
```
git clone https://github.com/MathRobotics/RoboKots
```

Install RoboKots
```
pip install .
```

## Examples

* `example/optimization_example/inverse_kinematics.py`: Uses RoboKots' kinematics
  and analytic Jacobian utilities in a simple Gauss-Newton inverse kinematics
  optimization.
* `example/optimization_example/liteopt_two_link.py`: Demonstrates solving a
  planar 2-link inverse-kinematics problem using LiteOpt's nonlinear least-
  squares solver with RoboKots-provided kinematics and analytic Jacobians.
* `example/optimization_example/trajectory_parameter_optimization.py`: Shows how
  to optimize polynomial trajectory parameters using inward/outward least-squares
  utilities. Run with ``python -m example.optimization_example.trajectory_parameter_optimization``
  from the repository root so imports resolve without manual path changes.

## Project Files (JSON)

RoboKots supports a simple project JSON format as the single entry point. The
project can either embed a simplified task or reference a full internal
`task.json`. When both are provided, it is treated as an error. Project-level
`time` overrides any task-level `time`.

Embedded task example:
```json
{
  "project": { "name": "ik_demo" },
  "model": { "path": "../model/2dof_arm.json" },
  "task": {
    "variables": { "q": [0.0, 0.0] },
    "targets": [
      { "type": "link_pos", "link": "arm2", "value": [2.2, 0.3, 0.0] }
    ]
  }
}
```

Task file reference example:
```json
{
  "project": { "name": "ik_demo_task_file" },
  "model": { "path": "../model/2dof_arm.json" },
  "task_file": "task.json",
  "time": { "N": 1, "dt": 0.01 }
}
```

Run the inverse kinematics example from the repository root:
```bash
python -m example.optimization_example.inverse_kinematics
```
