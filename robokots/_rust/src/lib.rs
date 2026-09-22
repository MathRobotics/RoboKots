#[cfg(feature = "python")]
use pyo3::exceptions::PyTypeError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

mod algorithms;
mod cmtm_generic;
mod cmtm_series;
mod dynamics_outputs;
pub mod error;
pub mod model;
mod pinocchio_like;
#[cfg(feature = "python")]
mod py_api;
mod rust_data;
mod spatial;
pub mod types;
mod workspace;

#[cfg(feature = "python")]
use py_api::{RustAbaData, RustBatchOutwardData, RustCompiledRobot, RustFastData, RustOutwardData, RustSelectedWorkspace};

#[cfg(feature = "python")]
#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustCompiledRobot>()?;
    m.add_class::<RustSelectedWorkspace>()?;
    m.add_class::<RustFastData>()?;
    m.add_class::<RustAbaData>()?;
    m.add_class::<RustOutwardData>()?;
    m.add_class::<RustBatchOutwardData>()?;
    if m.name()? != "robokots._rust_core" {
        return Err(PyTypeError::new_err("unexpected module name"));
    }
    Ok(())
}
