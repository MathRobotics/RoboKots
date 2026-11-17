"""Validated, optional native backend. Import only for explicit Rust use."""
from importlib import import_module

_NATIVE_API_VERSION = 1

try:
    _native = import_module("robokots_native")
except ModuleNotFoundError as exc:
    if exc.name != "robokots_native":
        raise ImportError("RoboKots native backend failed to load a dependency.") from exc
    raise ImportError(
        "RoboKots Rust backend requires the separately installed robokots-native "
        "package (native API 1). Install a compatible local native wheel, "
        "or select backend='numpy'."
    ) from exc
except ImportError as exc:
    raise ImportError("RoboKots native backend could not be loaded; check the platform and Python version of the native wheel.") from exc

_api = getattr(_native, "__api_version__", None)
if type(_api) is not int or _api != _NATIVE_API_VERSION:
    raise ImportError(
        f"Incompatible RoboKots native API: expected {_NATIVE_API_VERSION}, "
        f"got {_api!r}. Install a compatible robokots-native wheel."
    )

__all__ = ["RustBatchOutwardData", "RustCompiledRobot", "RustFastData",
           "RustOutwardData", "RustAbaData", "RustSelectedWorkspace"]
for _name in __all__:
    if not hasattr(_native, _name):
        raise ImportError(f"Incomplete RoboKots native API {_api}: missing {_name}.")
    globals()[_name] = getattr(_native, _name)
