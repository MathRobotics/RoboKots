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
        Self::new_impl(robot, true)
    }

    /// Kinematics and RNEA do not require the dense geometric Jacobian.
    /// Keeping this allocation out of ABA is important for large models:
    /// otherwise each one-shot ABA call allocates O(links * dof) unused data.
    pub(crate) fn new_without_jacobian(robot: &RustCompiledRobot) -> Self {
        Self::new_impl(robot, false)
    }

    fn new_impl(robot: &RustCompiledRobot, with_jacobian: bool) -> Self {
        Self {
            r: vec![0.0; robot.link_num * 9],
            p: vec![0.0; robot.link_num * 3],
            w: vec![0.0; robot.link_num * 3],
            lin_v: vec![0.0; robot.link_num * 3],
            alpha: vec![0.0; robot.link_num * 3],
            lin_a: vec![0.0; robot.link_num * 3],
            forces: vec![0.0; robot.link_num * 6],
            tau: vec![0.0; robot.dof],
            jac: if with_jacobian { vec![0.0; robot.link_num * 6 * robot.dof] } else { Vec::new() },
            active_axes: if with_jacobian { vec![[0.0; 3]; robot.dof] } else { Vec::new() },
            active_points: if with_jacobian { vec![[0.0; 3]; robot.dof] } else { Vec::new() },
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

/// Allocation-free order-zero articulated-body workspace.
///
/// Kept independent from `DynamicsCmtmWorkspace`: ABA's hot path must not
/// touch series/factorial buffers, while a future CMTM ABA will need series
/// articulated inertias rather than these scalar blocks.
pub(crate) struct AbaWorkspace {
    pub(crate) kinematics: Workspace,
    pub(crate) ia: Vec<[[f64; 6]; 6]>, pub(crate) pa: Vec<[f64; 6]>,
    pub(crate) c: Vec<[f64; 6]>, pub(crate) accel: Vec<[f64; 6]>,
    pub(crate) s: Vec<[f64; 6]>, pub(crate) u_vec: Vec<[f64; 6]>,
    pub(crate) u: Vec<f64>, pub(crate) d: Vec<f64>, pub(crate) qdd: Vec<f64>,
    pub(crate) bias_qdd: Vec<f64>,
    pub(crate) zero: Vec<f64>,
}

impl AbaWorkspace {
    pub(crate) fn new(robot: &RustCompiledRobot) -> Self {
        Self {
            kinematics: Workspace::new_without_jacobian(robot),
            ia: vec![[[0.0; 6]; 6]; robot.link_num], pa: vec![[0.0; 6]; robot.link_num],
            c: vec![[0.0; 6]; robot.link_num], accel: vec![[0.0; 6]; robot.link_num],
            s: vec![[0.0; 6]; robot.joint_num], u_vec: vec![[0.0; 6]; robot.joint_num],
            u: vec![0.0; robot.joint_num], d: vec![0.0; robot.joint_num],
            qdd: vec![0.0; robot.dof], bias_qdd: vec![0.0; robot.dof], zero: vec![0.0; robot.dof],
        }
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
    /// Heap capacity of numerical buffers, excluding structs and allocator overhead.
    pub(crate) fn buffer_bytes(&self) -> usize {
        self.link_mat.capacity() * std::mem::size_of::<f64>()
            + self.link_vecs.capacity() * std::mem::size_of::<f64>()
            + self.joint_mat.capacity() * std::mem::size_of::<f64>()
            + self.joint_vecs.capacity() * std::mem::size_of::<f64>()
            + self.fast_r.capacity() * std::mem::size_of::<f64>()
            + self.fast_p.capacity() * std::mem::size_of::<f64>()
            + self.fast_w.capacity() * std::mem::size_of::<f64>()
            + self.fast_lin_v.capacity() * std::mem::size_of::<f64>()
            + self.fast_alpha.capacity() * std::mem::size_of::<f64>()
            + self.fast_lin_a.capacity() * std::mem::size_of::<f64>()
            + self.factorial.capacity() * std::mem::size_of::<f64>()
            + self.tmp_rel_vecs.capacity() * std::mem::size_of::<f64>()
            + self.tmp_out_vecs.capacity() * std::mem::size_of::<f64>()
            + self.tmp_mat4_blocks_a.capacity() * std::mem::size_of::<[[f64; 4]; 4]>()
            + self.tmp_mat4_blocks_b.capacity() * std::mem::size_of::<[[f64; 4]; 4]>()
            + self.tmp_mat4_blocks_out.capacity() * std::mem::size_of::<[[f64; 4]; 4]>()
            + self.tmp_hat4_blocks.capacity() * std::mem::size_of::<[[f64; 4]; 4]>()
    }

    pub(crate) fn new(robot: &RustCompiledRobot, order: usize) -> Self {
        let mut factorial = vec![1.0; order.max(1)];
        crate::spatial::fill_factorial_table(&mut factorial);
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
            factorial,
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
    pub(crate) fn kinematics_only(robot: &RustCompiledRobot, order: usize, rhs_cols: usize) -> Self {
        Self {
            rhs_cols,
            link_mat: vec![0.0; robot.link_num * 16 * rhs_cols],
            link_vecs: vec![0.0; robot.link_num * (order - 1) * 6 * rhs_cols],
            joint_mat: vec![0.0; robot.joint_num * 16 * rhs_cols],
            joint_vecs: vec![0.0; robot.joint_num * (order - 1) * 6 * rhs_cols],
            link_momentum: Vec::new(), link_force: Vec::new(), joint_momentum: Vec::new(),
            joint_force: Vec::new(), joint_gravity_force: Vec::new(), joint_torque: Vec::new(),
        }
    }

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
    /// Heap capacity of numerical buffers, excluding structs and allocator overhead.
    pub(crate) fn buffer_bytes(&self) -> usize {
        self.link_momentum.capacity() * std::mem::size_of::<f64>()
            + self.link_force.capacity() * std::mem::size_of::<f64>()
            + self.joint_momentum.capacity() * std::mem::size_of::<f64>()
            + self.joint_force.capacity() * std::mem::size_of::<f64>()
            + self.joint_gravity_force.capacity() * std::mem::size_of::<f64>()
            + self.link_local_gravity.capacity() * std::mem::size_of::<f64>()
            + self.joint_torque.capacity() * std::mem::size_of::<f64>()
            + self.factorial.capacity() * std::mem::size_of::<f64>()
            + self.tmp_link_momentum.capacity() * std::mem::size_of::<f64>()
            + self.tmp_joint_momentum.capacity() * std::mem::size_of::<f64>()
            + self.tmp_force.capacity() * std::mem::size_of::<f64>()
            + self.tmp_gravity_force.capacity() * std::mem::size_of::<f64>()
            + self.tmp_local_gravity.capacity() * std::mem::size_of::<f64>()
            + self.tmp_rel_vecs.capacity() * std::mem::size_of::<f64>()
            + self.tmp_scaled_vecs.capacity() * std::mem::size_of::<f64>()
            + self.cached_motion.capacity() * std::mem::size_of::<f64>()
            + self.tmp_wrench_adj_a_blocks.capacity() * std::mem::size_of::<[[f64; 3]; 3]>()
            + self.tmp_wrench_adj_c_blocks.capacity() * std::mem::size_of::<[[f64; 3]; 3]>()
    }

    pub(crate) fn new(robot: &RustCompiledRobot, dynamics_order: usize) -> Self {
        let mut workspace = Self::kinematics_only(robot, dynamics_order + 2);
        workspace.ensure_dynamics(robot, dynamics_order);
        workspace
    }

    /// One shared kinematics state; dynamics-only arrays initially own no heap storage.
    pub(crate) fn kinematics_only(robot: &RustCompiledRobot, order: usize) -> Self {
        Self {
            cmtm: CmtmWorkspace::new(robot, order),
            link_momentum: Vec::new(),
            link_force: Vec::new(),
            joint_momentum: Vec::new(),
            joint_force: Vec::new(),
            joint_gravity_force: Vec::new(),
            link_local_gravity: Vec::new(),
            joint_torque: Vec::new(),
            factorial: Vec::new(),
            tmp_link_momentum: Vec::new(),
            tmp_joint_momentum: Vec::new(),
            tmp_force: Vec::new(),
            tmp_gravity_force: Vec::new(),
            tmp_local_gravity: Vec::new(),
            tmp_rel_vecs: Vec::new(),
            tmp_scaled_vecs: Vec::new(),
            cached_motion: Vec::new(),
            tmp_wrench_adj_a_blocks: Vec::new(),
            tmp_wrench_adj_c_blocks: Vec::new(),
        }
    }

    /// Allocate once on the first dynamics evaluation. Never replace the shared
    /// CMTM buffers or discard allocations when switching back to kinematics.
    pub(crate) fn ensure_dynamics(&mut self, robot: &RustCompiledRobot, dynamics_order: usize) {
        // A populated factorial table is the allocation marker, including zero-DOF models.
        if !self.factorial.is_empty() {
            return;
        }
        debug_assert_eq!(self.cmtm.factorial.len(), dynamics_order + 2);
        self.link_momentum = vec![0.0; robot.link_num * (dynamics_order + 1) * 6];
        self.link_force = vec![0.0; robot.link_num * dynamics_order * 6];
        self.joint_momentum = vec![0.0; robot.joint_num * (dynamics_order + 1) * 6];
        self.joint_force = vec![0.0; robot.joint_num * dynamics_order * 6];
        self.joint_gravity_force = vec![0.0; robot.joint_num * dynamics_order * 6];
        self.link_local_gravity = vec![0.0; robot.link_num * dynamics_order * 3];
        self.joint_torque = vec![0.0; robot.joint_num * dynamics_order];
        self.factorial = vec![1.0; (dynamics_order + 2).max(1)];
        self.tmp_link_momentum = vec![0.0; (dynamics_order + 1) * 6];
        self.tmp_joint_momentum = vec![0.0; (dynamics_order + 1) * 6];
        self.tmp_force = vec![0.0; dynamics_order * 6];
        self.tmp_gravity_force = vec![0.0; dynamics_order * 6];
        self.tmp_local_gravity = vec![0.0; dynamics_order * 3];
        self.tmp_rel_vecs = vec![0.0; dynamics_order * 6];
        self.tmp_scaled_vecs = vec![0.0; dynamics_order * 6];
        self.cached_motion = vec![0.0; robot.dof * (dynamics_order + 2)];
        self.tmp_wrench_adj_a_blocks = vec![[[0.0; 3]; 3]; dynamics_order + 1];
        self.tmp_wrench_adj_c_blocks = vec![[[0.0; 3]; 3]; dynamics_order + 1];
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

/// Order-zero RNEA linearization and reusable product scratch. Storage is
/// O(links + joints), independent of the number of generalized coordinates
/// squared. No motion Jacobian or basis seeds are stored.
pub(crate) struct RneaProductWorkspace {
    pub(crate) motion: Vec<f64>,
    pub(crate) gravity: [f64; 3],
    pub(crate) ready: bool,
    pub(crate) x: Vec<[[f64; 6]; 6]>,
    pub(crate) s: Vec<[f64; 6]>,
    pub(crate) v: Vec<[f64; 6]>,
    pub(crate) a: Vec<[f64; 6]>,
    pub(crate) f: Vec<[f64; 6]>,
    pub(crate) velocity_force: Vec<[[f64; 6]; 6]>,
    pub(crate) velocity_accel: Vec<[[f64; 6]; 6]>,
    pub(crate) q_velocity: Vec<[f64; 6]>,
    pub(crate) q_accel: Vec<[f64; 6]>,
    pub(crate) v_accel: Vec<[f64; 6]>,
    pub(crate) q_force: Vec<[f64; 6]>,
    pub(crate) dv: Vec<[f64; 6]>,
    pub(crate) da: Vec<[f64; 6]>,
    pub(crate) df: Vec<[f64; 6]>,
}

impl RneaProductWorkspace {
    pub(crate) fn new(robot: &RustCompiledRobot) -> Self {
        let n = robot.link_num;
        let j = robot.joint_num;
        Self {
            motion: Vec::new(),
            gravity: [0.; 3],
            ready: false,
            x: vec![[[0.; 6]; 6]; j],
            s: vec![[0.; 6]; j],
            v: vec![[0.; 6]; n],
            a: vec![[0.; 6]; n],
            f: vec![[0.; 6]; n],
            velocity_force: vec![[[0.; 6]; 6]; n],
            velocity_accel: vec![[[0.; 6]; 6]; j],
            q_velocity: vec![[0.; 6]; j],
            q_accel: vec![[0.; 6]; j],
            v_accel: vec![[0.; 6]; j],
            q_force: vec![[0.; 6]; j],
            dv: vec![[0.; 6]; n],
            da: vec![[0.; 6]; n],
            df: vec![[0.; 6]; n],
        }
    }
}
