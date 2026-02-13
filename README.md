# RoboKots

Utilities based on the Kots framework for robotic system modeling, kinematics, and dynamics analysis.

Clone RoboKots
```
git clone https://github.com/MathRobotics/RoboKots
```

Install RoboKots
```
pip install .
```

## Examples

* `example/simple_example/main.py`: End-to-end kinematics and dynamics Jacobian
  checks with a sample robot model.
* `example/polars_example/main.py`: Demonstrates how to export state to JSONL and
  use Polars to do fast, columnar analytics (feature extraction and filtering).

Run an example from the repository root:
```bash
python -m example.simple_example.main
```

## URDF Input

You can load URDF directly and reuse the same pipeline:
```python
from robokots.kots import Kots

kots = Kots.from_urdf_file("robot.urdf", order=3)
```
