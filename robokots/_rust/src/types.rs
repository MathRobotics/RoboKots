use pyo3::prelude::*;

use crate::pinocchio_like::PinocchioLikeWorkspace;
use crate::workspace::{CmtmWorkspace, DynamicsCmtmWorkspace};

#[derive(Clone)]
#[pyclass(name = "RustCompiledRobot")]
pub struct RustCompiledRobot {
    pub(crate) link_num: usize,
    pub(crate) joint_num: usize,
    pub(crate) dof: usize,
    pub(crate) parent_link: Vec<usize>,
    pub(crate) child_link: Vec<usize>,
    pub(crate) q_index: Vec<isize>,
    pub(crate) is_prismatic: Vec<bool>,
    pub(crate) axis: Vec<[f64; 3]>,
    pub(crate) origin_r: Vec<[[f64; 3]; 3]>,
    pub(crate) origin_p: Vec<[f64; 3]>,
    pub(crate) link_inertia: Vec<[[f64; 6]; 6]>,
    pub(crate) link_ancestors: Vec<Vec<usize>>,
    pub(crate) link_motion_columns: Vec<Vec<usize>>,
    pub(crate) link_subtree_motion_columns: Vec<Vec<usize>>,
    pub(crate) link_child_joints: Vec<Vec<usize>>,
}

#[pyclass(name = "RustFastData")]
pub struct RustFastData {
    pub(crate) robot: RustCompiledRobot,
    pub(crate) workspace: PinocchioLikeWorkspace,
    pub(crate) has_kinematics: bool,
    pub(crate) has_dynamics: bool,
    pub(crate) has_joint_jacobians: bool,
}

#[pyclass(name = "RustOutwardData")]
pub struct RustOutwardData {
    pub(crate) robot: RustCompiledRobot,
    pub(crate) order: usize,
    pub(crate) dynamics_order: usize,
    pub(crate) kinematics: CmtmWorkspace,
    pub(crate) dynamics: DynamicsCmtmWorkspace,
    pub(crate) has_kinematics: bool,
    pub(crate) has_dynamics: bool,
    pub(crate) has_cached_order1_dynamics: bool,
}

#[pyclass(name = "RustBatchOutwardData")]
pub struct RustBatchOutwardData {
    pub(crate) robot: RustCompiledRobot,
    pub(crate) order: usize,
    pub(crate) dynamics_order: usize,
    pub(crate) batch: usize,
    pub(crate) kinematics: Vec<CmtmWorkspace>,
    pub(crate) dynamics: Vec<DynamicsCmtmWorkspace>,
    pub(crate) has_kinematics: bool,
    pub(crate) has_dynamics: bool,
    pub(crate) has_cached_order1_dynamics: bool,
}
