"""Experimental Rust backend bindings for RoboKots."""

try:
    from robokots._rust_core import RustBatchOutwardData, RustCompiledRobot, RustFastData, RustOutwardData
except ImportError as exc:  # pragma: no cover - depends on local extension build
    raise ImportError(
        "RoboKots Rust backend is not built. Install RoboKots from the "
        "repository root with `pip install .` or `uv pip install .`. For "
        "Rust-only backend development, run "
        "`uvx maturin develop --manifest-path robokots/_rust/Cargo.toml`."
    ) from exc

__all__ = ["RustBatchOutwardData", "RustCompiledRobot", "RustFastData", "RustOutwardData"]
