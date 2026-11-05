"""Import boundaries for shared state containers and export consumers."""

import subprocess
import sys


def test_state_containers_and_exports_do_not_load_api_or_backends():
    subprocess.run(
        [sys.executable, "-c", """
import sys
from robokots.core.state import (
    StateType, StateBatch, StateTensor, JacobianTensor,
    OutwardDataView, StateValueProvider,
)
assert not any(name.startswith(('robokots.api', 'robokots.outward', 'mathrobo'))
               for name in sys.modules)
from robokots import core
for name in core.__all__:
    getattr(core, name)
assert not any(name.startswith(('robokots.api', 'robokots.outward'))
               for name in sys.modules)
from robokots.core.state import access
assert not any(name.startswith(('robokots.api', 'robokots.outward'))
               for name in sys.modules)
from robokots.state_io.dictionary import export_state_dict
from robokots.outward.data import OutwardState, ArrayOutwardState
assert not any(name.startswith('robokots.api') for name in sys.modules)
"""],
        check=True,
    )


def test_core_exports_shared_state_containers():
    from robokots import core
    from robokots.core.state import StateBatch, StateTensor, JacobianTensor

    for cls in (StateBatch, StateTensor, JacobianTensor):
        assert getattr(core, cls.__name__) is cls
