//! Native robot models, states and direct Jacobian products. Python is optional.
//!
//! ```
//! use robokots_rust::types::RustCompiledRobot;
//! use robokots_rust::{StateOwner, StateQuantity, StateOutput, ReferenceFrame};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let robot = RustCompiledRobot::from_json(r#"{
//!   "schema_version":"0.0.2",
//!   "links":[{"id":0,"name":"world"},{"id":1,"name":"arm","mass":1}],
//!   "joints":[{"id":0,"name":"axis","type":"revolute",
//!              "parent_link_id":0,"child_link_id":1,"axis":[0,0,1]}]
//! }"#, true)?;
//! let mut state = robot.create_outward_data(2)?;
//! state.compute_kinematics(&[0.2, 0.3])?;
//! let world_velocity = state.world_link_vec(1, 2)?;
//! assert!((world_velocity[2] - 0.3).abs() < 1e-12);
//! let output = StateOutput::new(StateOwner::Link(1),
//!     StateQuantity::SpatialMotion, 0, ReferenceFrame::World);
//! let mut products = robot.create_selected_workspace(2)?;
//! let jvp = products.apply(&[0.2, 0.3], &[0., 1.], &[output], 1, 1, [0.;3], false)?;
//! assert!((jvp[2] - 1.).abs() < 1e-12);
//! let tau = robot.inverse_dynamics(&[0.2], &[0.3], &[0.4], [0.,0.,-9.81])?;
//! let acceleration = robot.forward_dynamics(&[0.2], &[0.3], &tau, [0.,0.,-9.81])?;
//! assert!((acceleration[0] - 0.4).abs() < 1e-12);
//! # Ok(())
//! # }
//! ```

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
mod model_input;
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

/// Typed native selected-output descriptors used by JVP/VJP.
pub use dynamics_outputs::{StateOwner, StateQuantity, ReferenceFrame, StateOutput};
