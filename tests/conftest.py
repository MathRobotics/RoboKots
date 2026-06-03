from pathlib import Path
import sys

import jax


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
