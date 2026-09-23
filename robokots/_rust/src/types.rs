use crate::pinocchio_like::PinocchioLikeWorkspace;
use crate::workspace::{AbaWorkspace, DynamicsCmtmWorkspace};

#[derive(Clone)]
pub struct RustCompiledRobot {
    pub(crate) info: std::sync::Arc<crate::model::ModelInfo>,
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
    /// Link indices in each link's subtree, in forward-topology order.
    /// Unlike `link_subtree_motion_columns`, this includes fixed links.
    pub(crate) link_subtree_links: Vec<Vec<usize>>,
    pub(crate) link_child_joints: Vec<Vec<usize>>,
}

pub struct RustFastData {
    pub(crate) robot: RustCompiledRobot,
    pub(crate) workspace: PinocchioLikeWorkspace,
    pub(crate) has_kinematics: bool,
    pub(crate) has_dynamics: bool,
    pub(crate) has_joint_jacobians: bool,
}

/// Reusable storage for the order-zero articulated-body algorithm.
///
/// This deliberately owns a scalar [`AbaWorkspace`] rather than any CMTM
/// buffers.  A future CMTM ABA data object will have series-valued articulated
/// quantities and can share topology/spatial primitives without making the
/// scalar hot path pay for those buffers.
pub struct RustAbaData {
    pub(crate) robot: RustCompiledRobot,
    pub(crate) workspace: AbaWorkspace,
    pub(crate) factor_q: Vec<f64>,
    pub(crate) bias_q: Vec<f64>,
    pub(crate) bias_v: Vec<f64>,
    pub(crate) bias_gravity: [f64; 3],
    pub(crate) prepared: bool,
}

pub struct RustOutwardData {
    pub(crate) robot: RustCompiledRobot,
    pub(crate) order: usize,
    pub(crate) dynamics_order: usize,
    pub(crate) dynamics: DynamicsCmtmWorkspace,
    pub(crate) has_kinematics: bool,
    pub(crate) has_dynamics: bool,
    pub(crate) has_cached_order1_dynamics: bool,
    pub(crate) has_full_dynamics: bool,
}

pub struct RustBatchOutwardData {
    pub(crate) robot: RustCompiledRobot,
    pub(crate) order: usize,
    pub(crate) dynamics_order: usize,
    pub(crate) batch: usize,
    pub(crate) dynamics: Vec<DynamicsCmtmWorkspace>,
    pub(crate) has_kinematics: bool,
    pub(crate) has_dynamics: bool,
    pub(crate) has_cached_order1_dynamics: bool,
    pub(crate) has_full_dynamics: bool,
}


/// Bounded latest-batch cache of derivative primals and a reusable tangent buffer.
/// Independent of semantic Python StateCache; owned by one compiled model.
pub struct RustSelectedWorkspace {
    pub(crate) robot: RustCompiledRobot,
    pub(crate) order: usize,
    pub(crate) primal: Vec<DynamicsCmtmWorkspace>,
    pub(crate) tangent: Option<crate::workspace::DynamicsCmtmTangentWorkspace>,
    pub(crate) rnea_products: Vec<crate::workspace::RneaProductWorkspace>,
    pub(crate) route_cache: Vec<Option<crate::dynamics_outputs::KinematicRouteWorkspace>>,
    pub(crate) motion: Vec<f64>,
    pub(crate) gravity: [f64;3],
    pub(crate) dynamic: bool,
    pub(crate) ready: bool,
    pub(crate) kinematics_evaluations: usize,
    pub(crate) dynamics_evaluations: usize,
}

use crate::error::{CoreResult, Error};

impl RustCompiledRobot {
    pub fn create_selected_workspace(&self, order: usize) -> CoreResult<crate::types::RustSelectedWorkspace> {
        self.check_cmtm_supported()?;
        if order == 0 { return Err(Error::new("selected workspace order must be positive")); }
        Ok(crate::types::RustSelectedWorkspace {
            robot: self.clone(), order, primal: Vec::new(), tangent: None, motion: Vec::new(),
            rnea_products: Vec::new(), route_cache: Vec::new(),
            gravity: [0.0;3], dynamic: false, ready: false,
            kinematics_evaluations: 0, dynamics_evaluations: 0,
        })
    }

    pub fn create_outward_data(&self, order: usize) -> CoreResult<RustOutwardData> {
        self.check_cmtm_supported()?;
        if order < 1 {
            return Err(Error::new("order must be >= 1"));
        }
        let dynamics_order = order.saturating_sub(2);
        Ok(RustOutwardData {
            robot: self.clone(),
            order,
            dynamics_order,
            dynamics: DynamicsCmtmWorkspace::kinematics_only(self, order),
            has_kinematics: false,
            has_dynamics: false,
            has_cached_order1_dynamics: false,
            has_full_dynamics: false,
        })
    }

    pub fn create_fast_data(&self) -> RustFastData {
        self.create_pinocchio_like_data()
    }

    pub fn create_aba_data(&self) -> RustAbaData {
        RustAbaData {
            robot: self.clone(), workspace: AbaWorkspace::new(self),
            factor_q: Vec::new(), bias_q: Vec::new(), bias_v: Vec::new(),
            bias_gravity: [0.0; 3], prepared: false,
        }
    }

    pub fn create_pinocchio_like_data(&self) -> RustFastData {
        RustFastData {
            robot: self.clone(),
            workspace: PinocchioLikeWorkspace::new(self),
            has_kinematics: false,
            has_dynamics: false,
            has_joint_jacobians: false,
        }
    }

    pub fn create_batch_outward_data(
        &self,
        order: usize,
        batch: usize,
    ) -> CoreResult<RustBatchOutwardData> {
        self.check_cmtm_supported()?;
        if order < 1 {
            return Err(Error::new("order must be >= 1"));
        }
        let dynamics_order = order.saturating_sub(2);
        let mut dynamics = Vec::with_capacity(batch);
        for _ in 0..batch {
            dynamics.push(DynamicsCmtmWorkspace::kinematics_only(self, order));
        }
        Ok(RustBatchOutwardData {
            robot: self.clone(),
            order,
            dynamics_order,
            batch,
            dynamics,
            has_kinematics: false,
            has_dynamics: false,
            has_cached_order1_dynamics: false,
            has_full_dynamics: false,
        })
    }
}
