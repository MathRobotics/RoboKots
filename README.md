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
* `example/polars_example/main.py`: Demonstrates how to export state to JSONL and
  use Polars to do fast, columnar analytics (feature extraction and filtering).
