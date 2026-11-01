//! Python extension bridge to the independently developed native core.
use pyo3::prelude::*;
#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    robokots::register_python_module(m)
}
