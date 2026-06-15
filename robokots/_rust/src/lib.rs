use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

mod algorithms;
mod cmtm_generic;
mod cmtm_series;
mod model;
mod pinocchio_like;
mod py_api;
mod rust_data;
mod spatial;
mod types;
mod workspace;

use types::{RustBatchOutwardData, RustCompiledRobot, RustFastData, RustOutwardData};

#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustCompiledRobot>()?;
    m.add_class::<RustFastData>()?;
    m.add_class::<RustOutwardData>()?;
    m.add_class::<RustBatchOutwardData>()?;
    if m.name()? != "robokots._rust_core" {
        return Err(PyTypeError::new_err("unexpected module name"));
    }
    Ok(())
}
