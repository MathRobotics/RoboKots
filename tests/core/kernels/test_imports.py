"""Kernel imports must not load optional backends or API orchestration."""

import subprocess
import sys


def test_numpy_kernels_and_lazy_whole_body_exports_are_independent():
    subprocess.run([sys.executable, "-c", """
import sys
from robokots.core.kernels import (
    joint, inertia, dynamics, dynamics_derivatives, cmtm_apply, whole_body,
)
for name in whole_body.__all__:
    getattr(whole_body, name)
assert not any(name.startswith(('robokots.core.kernels.kinematics_jax',
                               'robokots.core.kernels.soft_link',
                               'robokots.api', 'robokots.outward', 'robokots.core.models'))
               for name in sys.modules)
"""], check=True)
