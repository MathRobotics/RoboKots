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
* `example/optimization_example/trajectory_parameter_optimization.py`: Shows how
  to optimize polynomial trajectory parameters using inward/outward least-squares
  utilities. Run with ``python -m example.optimization_example.trajectory_parameter_optimization``
  from the repository root so imports resolve without manual path changes.
