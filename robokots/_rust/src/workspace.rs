use crate::types::RustCompiledRobot;

pub(crate) struct Workspace {
    pub(crate) r: Vec<f64>,
    pub(crate) p: Vec<f64>,
    pub(crate) w: Vec<f64>,
    pub(crate) lin_v: Vec<f64>,
    pub(crate) alpha: Vec<f64>,
    pub(crate) lin_a: Vec<f64>,
    pub(crate) forces: Vec<f64>,
    pub(crate) tau: Vec<f64>,
    pub(crate) jac: Vec<f64>,
    pub(crate) active_axes: Vec<[f64; 3]>,
    pub(crate) active_points: Vec<[f64; 3]>,
    pub(crate) zero_motion: Vec<f64>,
}

impl Workspace {
    pub(crate) fn new(robot: &RustCompiledRobot) -> Self {
        Self {
            r: vec![0.0; robot.link_num * 9],
            p: vec![0.0; robot.link_num * 3],
            w: vec![0.0; robot.link_num * 3],
            lin_v: vec![0.0; robot.link_num * 3],
            alpha: vec![0.0; robot.link_num * 3],
            lin_a: vec![0.0; robot.link_num * 3],
            forces: vec![0.0; robot.link_num * 6],
            tau: vec![0.0; robot.dof],
            jac: vec![0.0; robot.link_num * 6 * robot.dof],
            active_axes: vec![[0.0; 3]; robot.dof],
            active_points: vec![[0.0; 3]; robot.dof],
            zero_motion: vec![0.0; robot.dof],
        }
    }

    pub(crate) fn clear_kinematics(&mut self) {
        self.r.fill(0.0);
        self.p.fill(0.0);
        self.w.fill(0.0);
        self.lin_v.fill(0.0);
        self.alpha.fill(0.0);
        self.lin_a.fill(0.0);
    }
}

pub(crate) struct BulkDerivativeWorkspace {
    pub(crate) cols: usize,
    pub(crate) r: Vec<f64>,
    pub(crate) p: Vec<f64>,
    pub(crate) w: Vec<f64>,
    pub(crate) lin_v: Vec<f64>,
    pub(crate) alpha: Vec<f64>,
    pub(crate) lin_a: Vec<f64>,
    pub(crate) forces: Vec<f64>,
    pub(crate) tau: Vec<f64>,
}

impl BulkDerivativeWorkspace {
    pub(crate) fn new(robot: &RustCompiledRobot, cols: usize) -> Self {
        Self {
            cols,
            r: vec![0.0; robot.link_num * 9 * cols],
            p: vec![0.0; robot.link_num * 3 * cols],
            w: vec![0.0; robot.link_num * 3 * cols],
            lin_v: vec![0.0; robot.link_num * 3 * cols],
            alpha: vec![0.0; robot.link_num * 3 * cols],
            lin_a: vec![0.0; robot.link_num * 3 * cols],
            forces: vec![0.0; robot.link_num * 6 * cols],
            tau: vec![0.0; robot.dof * cols],
        }
    }

    pub(crate) fn clear(&mut self) {
        self.r.fill(0.0);
        self.p.fill(0.0);
        self.w.fill(0.0);
        self.lin_v.fill(0.0);
        self.alpha.fill(0.0);
        self.lin_a.fill(0.0);
        self.forces.fill(0.0);
        self.tau.fill(0.0);
    }
}

pub(crate) struct CmtmWorkspace {
    pub(crate) link_mat: Vec<f64>,
    pub(crate) link_vecs: Vec<f64>,
    pub(crate) joint_mat: Vec<f64>,
    pub(crate) joint_vecs: Vec<f64>,
    pub(crate) fast_r: Vec<f64>,
    pub(crate) fast_p: Vec<f64>,
    pub(crate) fast_w: Vec<f64>,
    pub(crate) fast_lin_v: Vec<f64>,
    pub(crate) fast_alpha: Vec<f64>,
    pub(crate) fast_lin_a: Vec<f64>,
    pub(crate) factorial: Vec<f64>,
    pub(crate) tmp_rel_vecs: Vec<f64>,
    pub(crate) tmp_out_vecs: Vec<f64>,
    pub(crate) tmp_mat4_blocks_a: Vec<[[f64; 4]; 4]>,
    pub(crate) tmp_mat4_blocks_b: Vec<[[f64; 4]; 4]>,
    pub(crate) tmp_mat4_blocks_out: Vec<[[f64; 4]; 4]>,
    pub(crate) tmp_hat4_blocks: Vec<[[f64; 4]; 4]>,
}

impl CmtmWorkspace {
    pub(crate) fn new(robot: &RustCompiledRobot, order: usize) -> Self {
        Self {
            link_mat: vec![0.0; robot.link_num * 16],
            link_vecs: vec![0.0; robot.link_num * (order - 1) * 6],
            joint_mat: vec![0.0; robot.joint_num * 16],
            joint_vecs: vec![0.0; robot.joint_num * (order - 1) * 6],
            fast_r: vec![0.0; robot.link_num * 9],
            fast_p: vec![0.0; robot.link_num * 3],
            fast_w: vec![0.0; robot.link_num * 3],
            fast_lin_v: vec![0.0; robot.link_num * 3],
            fast_alpha: vec![0.0; robot.link_num * 3],
            fast_lin_a: vec![0.0; robot.link_num * 3],
            factorial: vec![1.0; order.max(1)],
            tmp_rel_vecs: vec![0.0; (order - 1) * 6],
            tmp_out_vecs: vec![0.0; (order - 1) * 6],
            tmp_mat4_blocks_a: vec![[[0.0; 4]; 4]; order],
            tmp_mat4_blocks_b: vec![[[0.0; 4]; 4]; order],
            tmp_mat4_blocks_out: vec![[[0.0; 4]; 4]; order],
            tmp_hat4_blocks: vec![[[0.0; 4]; 4]; order],
        }
    }

    pub(crate) fn clear(&mut self) {
        self.link_mat.fill(0.0);
        self.link_vecs.fill(0.0);
        self.joint_mat.fill(0.0);
        self.joint_vecs.fill(0.0);
        self.fast_r.fill(0.0);
        self.fast_p.fill(0.0);
        self.fast_w.fill(0.0);
        self.fast_lin_v.fill(0.0);
        self.fast_alpha.fill(0.0);
        self.fast_lin_a.fill(0.0);
        self.tmp_rel_vecs.fill(0.0);
        self.tmp_out_vecs.fill(0.0);
    }
}

pub(crate) struct DynamicsCmtmWorkspace {
    pub(crate) cmtm: CmtmWorkspace,
    pub(crate) link_momentum: Vec<f64>,
    pub(crate) link_force: Vec<f64>,
    pub(crate) joint_momentum: Vec<f64>,
    pub(crate) joint_force: Vec<f64>,
    pub(crate) joint_gravity_force: Vec<f64>,
    /// World gravity expressed in each link frame and its time derivatives.
    ///
    /// This is primal data for the CMTM reverse pass.  Keeping it per-link is
    /// important: the old `tmp_local_gravity` scratch is overwritten while
    /// walking the tree and therefore cannot be used by a later VJP.
    pub(crate) link_local_gravity: Vec<f64>,
    pub(crate) joint_torque: Vec<f64>,
    pub(crate) factorial: Vec<f64>,
    pub(crate) tmp_link_momentum: Vec<f64>,
    pub(crate) tmp_joint_momentum: Vec<f64>,
    pub(crate) tmp_force: Vec<f64>,
    pub(crate) tmp_gravity_force: Vec<f64>,
    pub(crate) tmp_local_gravity: Vec<f64>,
    pub(crate) tmp_rel_vecs: Vec<f64>,
    pub(crate) tmp_scaled_vecs: Vec<f64>,
    pub(crate) cached_motion: Vec<f64>,
    pub(crate) tmp_wrench_adj_a_blocks: Vec<[[f64; 3]; 3]>,
    pub(crate) tmp_wrench_adj_c_blocks: Vec<[[f64; 3]; 3]>,
}

/// Cotangents for the complete CMTM inverse-dynamics recurrence.
///
/// Layout matches [`DynamicsCmtmTangentWorkspace`]: for every primal scalar,
/// `rhs_cols` cotangents are contiguous.  A single workspace consequently
/// supports the IOC use-case of several output cotangents per trajectory
/// frame without materialising a dense Jacobian.
#[allow(dead_code)]
pub(crate) struct DynamicsCmtmReverseWorkspace {
    pub(crate) rhs_cols: usize,
    pub(crate) link_mat: Vec<f64>,
    pub(crate) link_vecs: Vec<f64>,
    pub(crate) joint_mat: Vec<f64>,
    pub(crate) joint_vecs: Vec<f64>,
    pub(crate) link_momentum: Vec<f64>,
    pub(crate) link_force: Vec<f64>,
    pub(crate) joint_momentum: Vec<f64>,
    pub(crate) joint_force: Vec<f64>,
    pub(crate) joint_gravity_force: Vec<f64>,
    pub(crate) link_local_gravity: Vec<f64>,
    pub(crate) joint_torque: Vec<f64>,
    /// Final cotangent in the scalar-major motion layout accepted by CMTM.
    pub(crate) motion: Vec<f64>,
}

#[allow(dead_code)]
impl DynamicsCmtmReverseWorkspace {
    pub(crate) fn new(
        robot: &RustCompiledRobot,
        dynamics_order: usize,
        rhs_cols: usize,
    ) -> Self {
        let kin_order = dynamics_order + 2;
        Self {
            rhs_cols,
            link_mat: vec![0.0; robot.link_num * 16 * rhs_cols],
            link_vecs: vec![0.0; robot.link_num * (kin_order - 1) * 6 * rhs_cols],
            joint_mat: vec![0.0; robot.joint_num * 16 * rhs_cols],
            joint_vecs: vec![0.0; robot.joint_num * (kin_order - 1) * 6 * rhs_cols],
            link_momentum: vec![0.0; robot.link_num * (dynamics_order + 1) * 6 * rhs_cols],
            link_force: vec![0.0; robot.link_num * dynamics_order * 6 * rhs_cols],
            joint_momentum: vec![0.0; robot.joint_num * (dynamics_order + 1) * 6 * rhs_cols],
            joint_force: vec![0.0; robot.joint_num * dynamics_order * 6 * rhs_cols],
            joint_gravity_force: vec![0.0; robot.joint_num * dynamics_order * 6 * rhs_cols],
            link_local_gravity: vec![0.0; robot.link_num * dynamics_order * 3 * rhs_cols],
            joint_torque: vec![0.0; robot.joint_num * dynamics_order * rhs_cols],
            motion: vec![0.0; robot.dof * kin_order * rhs_cols],
        }
    }

    pub(crate) fn clear(&mut self) {
        self.link_mat.fill(0.0);
        self.link_vecs.fill(0.0);
        self.joint_mat.fill(0.0);
        self.joint_vecs.fill(0.0);
        self.link_momentum.fill(0.0);
        self.link_force.fill(0.0);
        self.joint_momentum.fill(0.0);
        self.joint_force.fill(0.0);
        self.joint_gravity_force.fill(0.0);
        self.link_local_gravity.fill(0.0);
        self.joint_torque.fill(0.0);
        self.motion.fill(0.0);
    }
}

/// Directional derivatives of the CMTM inverse-dynamics recurrence.
///
/// Every buffer is laid out with `rhs_cols` contiguous tangent components per
/// primal scalar.  This is deliberately separate from `BulkDerivativeWorkspace`:
/// the latter differentiates the order-3 RNEA recurrence, while this workspace
/// retains the complete CMTM series needed for `torque_diff1` and higher.
#[allow(dead_code)]
pub(crate) struct DynamicsCmtmTangentWorkspace {
    pub(crate) rhs_cols: usize,
    pub(crate) link_mat: Vec<f64>,
    pub(crate) link_vecs: Vec<f64>,
    pub(crate) joint_mat: Vec<f64>,
    pub(crate) joint_vecs: Vec<f64>,
    pub(crate) link_momentum: Vec<f64>,
    pub(crate) link_force: Vec<f64>,
    pub(crate) joint_momentum: Vec<f64>,
    pub(crate) joint_force: Vec<f64>,
    pub(crate) joint_gravity_force: Vec<f64>,
    pub(crate) joint_torque: Vec<f64>,
}

#[allow(dead_code)]
impl DynamicsCmtmTangentWorkspace {
    pub(crate) fn new(
        robot: &RustCompiledRobot,
        dynamics_order: usize,
        rhs_cols: usize,
    ) -> Self {
        let kin_order = dynamics_order + 2;
        Self {
            rhs_cols,
            link_mat: vec![0.0; robot.link_num * 16 * rhs_cols],
            link_vecs: vec![0.0; robot.link_num * (kin_order - 1) * 6 * rhs_cols],
            joint_mat: vec![0.0; robot.joint_num * 16 * rhs_cols],
            joint_vecs: vec![0.0; robot.joint_num * (kin_order - 1) * 6 * rhs_cols],
            link_momentum: vec![0.0; robot.link_num * (dynamics_order + 1) * 6 * rhs_cols],
            link_force: vec![0.0; robot.link_num * dynamics_order * 6 * rhs_cols],
            joint_momentum: vec![0.0; robot.joint_num * (dynamics_order + 1) * 6 * rhs_cols],
            joint_force: vec![0.0; robot.joint_num * dynamics_order * 6 * rhs_cols],
            joint_gravity_force: vec![0.0; robot.joint_num * dynamics_order * 6 * rhs_cols],
            joint_torque: vec![0.0; robot.joint_num * dynamics_order * rhs_cols],
        }
    }

    pub(crate) fn clear(&mut self) {
        self.link_mat.fill(0.0);
        self.link_vecs.fill(0.0);
        self.joint_mat.fill(0.0);
        self.joint_vecs.fill(0.0);
        self.link_momentum.fill(0.0);
        self.link_force.fill(0.0);
        self.joint_momentum.fill(0.0);
        self.joint_force.fill(0.0);
        self.joint_gravity_force.fill(0.0);
        self.joint_torque.fill(0.0);
    }
}

impl DynamicsCmtmWorkspace {
    pub(crate) fn new(robot: &RustCompiledRobot, dynamics_order: usize) -> Self {
        Self {
            cmtm: CmtmWorkspace::new(robot, dynamics_order + 2),
            link_momentum: vec![0.0; robot.link_num * (dynamics_order + 1) * 6],
            link_force: vec![0.0; robot.link_num * dynamics_order * 6],
            joint_momentum: vec![0.0; robot.joint_num * (dynamics_order + 1) * 6],
            joint_force: vec![0.0; robot.joint_num * dynamics_order * 6],
            joint_gravity_force: vec![0.0; robot.joint_num * dynamics_order * 6],
            link_local_gravity: vec![0.0; robot.link_num * dynamics_order * 3],
            joint_torque: vec![0.0; robot.joint_num * dynamics_order],
            factorial: vec![1.0; (dynamics_order + 2).max(1)],
            tmp_link_momentum: vec![0.0; (dynamics_order + 1) * 6],
            tmp_joint_momentum: vec![0.0; (dynamics_order + 1) * 6],
            tmp_force: vec![0.0; dynamics_order * 6],
            tmp_gravity_force: vec![0.0; dynamics_order * 6],
            tmp_local_gravity: vec![0.0; dynamics_order * 3],
            tmp_rel_vecs: vec![0.0; dynamics_order * 6],
            tmp_scaled_vecs: vec![0.0; dynamics_order * 6],
            cached_motion: vec![0.0; robot.dof * (dynamics_order + 2)],
            tmp_wrench_adj_a_blocks: vec![[[0.0; 3]; 3]; dynamics_order + 1],
            tmp_wrench_adj_c_blocks: vec![[[0.0; 3]; 3]; dynamics_order + 1],
        }
    }

    pub(crate) fn clear(&mut self) {
        self.cmtm.clear();
        self.link_momentum.fill(0.0);
        self.link_force.fill(0.0);
        self.joint_momentum.fill(0.0);
        self.joint_force.fill(0.0);
        self.joint_gravity_force.fill(0.0);
        self.link_local_gravity.fill(0.0);
        self.joint_torque.fill(0.0);
        self.tmp_link_momentum.fill(0.0);
        self.tmp_joint_momentum.fill(0.0);
        self.tmp_force.fill(0.0);
        self.tmp_gravity_force.fill(0.0);
        self.tmp_local_gravity.fill(0.0);
        self.tmp_rel_vecs.fill(0.0);
        self.tmp_scaled_vecs.fill(0.0);
        self.cached_motion.fill(0.0);
    }

    pub(crate) fn clear_minimal(&mut self) {
        self.cmtm.clear();
        self.link_force.fill(0.0);
        self.joint_momentum.fill(0.0);
        self.joint_torque.fill(0.0);
        self.tmp_link_momentum.fill(0.0);
        self.tmp_joint_momentum.fill(0.0);
        self.tmp_force.fill(0.0);
        self.joint_gravity_force.fill(0.0);
        self.link_local_gravity.fill(0.0);
        self.tmp_gravity_force.fill(0.0);
        self.tmp_local_gravity.fill(0.0);
        self.tmp_rel_vecs.fill(0.0);
        self.tmp_scaled_vecs.fill(0.0);
        self.cached_motion.fill(0.0);
    }
}
