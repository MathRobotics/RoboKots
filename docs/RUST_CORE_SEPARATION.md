# Optional native backend

RoboKots contains the Python API, NumPy/JAX implementations and the adapter for
an optional native extension. Installing or building this repository requires
no Rust compiler, native source repository, credentials or sibling directory.

```bash
pip install .
```

Select `backend="numpy"` explicitly for NumPy-only operation. Native execution
is opt-in via `backend="rust"`. The Python API and array layouts are unchanged.

## Local native installation

Native source and build configuration are managed independently. Obtain a
compatible `robokots-native` wheel from its maintainer and install it into the
same environment as RoboKots:

```bash
python -m pip install /path/to/robokots_native-<version>-<platform-tags>.whl
```

Replace the placeholder with the actual wheel filename for your Python, OS and
CPU. This is a local installation procedure, not a public package-index upload.
No Rust compiler is needed to install the wheel.

The distribution is `robokots-native`, and its import name is `robokots_native`.
It must declare integer `__api_version__ = 1`. RoboKots checks this version and
required classes on first native use. Missing or incompatible extensions raise
ImportError with a reason; they never silently fall back to NumPy. Native API
version is a compatibility contract, separate from package release version.
Breaking signatures, array layouts or semantics require a new API version.

```python
from robokots.kots import Kots
model = Kots.from_urdf_file("robot.urdf", order=3, backend="rust")
```

`robokots._rust` remains the internal adapter. The old private import
`robokots._rust_core` is a Python compatibility shim, not a bundled binary.
The new native package owns no files under the `robokots` package, so installing
or uninstalling it cannot overwrite RoboKots Python modules.

## Migration and verification

Replace the old combined RoboKots installation with the Python package, then
install the separate native wheel. Remove any manually copied old
`robokots/_rust_core*.so` or `.pyd` from a source checkout: extension modules
can shadow the compatibility shim. Restart Python/Notebook processes afterward.
Do not copy compiled files into the public Git repository.

```python
import robokots
import robokots_native
from robokots._rust import RustCompiledRobot
print(robokots.__file__)
print(robokots_native.__file__)
print(robokots_native.__api_version__)
```

Public builds use setuptools (`uv build` or `python -m build`) and produce a
`py3-none-any` wheel and a source archive without native code. Native wheels are
built independently. Neither build is published automatically. Existing Git
history and previously distributed artifacts are not rewritten by this change.
