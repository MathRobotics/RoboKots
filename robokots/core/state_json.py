"""Compatibility alias for :mod:`robokots.core.state_jsonl`.

New code should import the implementation module directly. Both paths share
the same module object, including classes and mutable module-level caches.
"""

import sys as _sys

from . import state_jsonl as _implementation

_sys.modules[__name__] = _implementation
