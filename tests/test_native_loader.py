"""Native installation is optional, separately owned and API-version checked."""
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(script):
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = subprocess.run([sys.executable, "-c", script], env=env, cwd=ROOT,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_native_has_actionable_error_and_numpy_still_loads():
    run(r'''
import sys, importlib.abc
class MissingNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "robokots_native":
            raise ModuleNotFoundError("blocked", name=fullname)
sys.meta_path.insert(0, MissingNative())
from robokots.kots import Kots
model = {"schema_version":"0.0.2", "links":[{"id":0,"name":"world"}], "joints":[]}
Kots.from_json_data(model, backend="numpy")
try:
    Kots.from_json_data(model, backend="rust")
except ImportError as exc:
    assert "robokots-native" in str(exc) and "backend='numpy'" in str(exc)
else:
    raise AssertionError("Rust request silently succeeded")
''')


@pytest.mark.parametrize("version", [None, 0, 2, "1", True])
def test_native_api_mismatch(version):
    run(f'''
import sys, types
native = types.ModuleType("robokots_native")
native.__api_version__ = {version!r}
sys.modules["robokots_native"] = native
try:
    import robokots._rust
except ImportError as exc:
    assert "Incompatible RoboKots native API" in str(exc)
else:
    raise AssertionError("incompatible API accepted")
''')


def test_incomplete_native_rejected():
    run(r'''
import sys, types
native = types.ModuleType("robokots_native")
native.__api_version__ = 1
sys.modules["robokots_native"] = native
try:
    import robokots._rust
except ImportError as exc:
    assert "missing RustBatchOutwardData" in str(exc)
else:
    raise AssertionError("incomplete native accepted")
''')


def test_legacy_import_uses_validated_native_api():
    run(r'''
import sys, types
native = types.ModuleType("robokots_native")
native.__api_version__ = 1
for name in ("RustBatchOutwardData", "RustCompiledRobot", "RustFastData", "RustOutwardData", "RustAbaData", "RustSelectedWorkspace"):
    setattr(native, name, type(name, (), {}))
sys.modules["robokots_native"] = native
from robokots._rust import RustCompiledRobot
from robokots._rust_core import RustCompiledRobot as legacy
assert RustCompiledRobot is legacy is native.RustCompiledRobot
''')
