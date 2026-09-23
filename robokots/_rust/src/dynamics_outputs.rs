//! Selected dynamics, local/world spatial motion and pose share one primal and
//! derivative recurrence. Neither product kernel forms a dense motion Jacobian.
use crate::error::{Error, CoreResult};

use crate::spatial::*;
use crate::types::RustCompiledRobot;
use crate::workspace::{CmtmWorkspace, DynamicsCmtmTangentWorkspace, DynamicsCmtmWorkspace};

/// (owner: link=0/joint=1, owner index, momentum=0/force=1/torque=2/kinematics=3,
/// ordinary time derivative index, world frame).
/// Kinematics index zero denotes velocity; pose families 4/5/6 are pos/rot/frame.
/// Pose derivatives use body tangents by default and spatial tangents in world.
pub(crate) type DynamicsOutput = (usize, usize, usize, usize, bool);

/// Owner of an output; a joint motion is relative, not its child's absolute motion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateOwner { Link(usize), Joint(usize) }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateQuantity { Momentum, Force, Torque, SpatialMotion, Position, Rotation, Frame }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceFrame { Local, World }

/// Selected derivative output. `derivative` is the ordinary time derivative index:
/// SpatialMotion 0 is velocity, 1 acceleration. Pose quantities require index 0.
/// Pose Jacobians use body tangents (Local) or spatial tangents (World), rather
/// than derivatives of flattened rotation/frame matrix elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateOutput {
    pub owner: StateOwner,
    pub quantity: StateQuantity,
    pub derivative: usize,
    pub frame: ReferenceFrame,
}

impl StateOutput {
    pub const fn new(owner: StateOwner, quantity: StateQuantity, derivative: usize, frame: ReferenceFrame) -> Self {
        Self { owner, quantity, derivative, frame }
    }

    /// Number of tangent rows (rotation: 3, frame: 6, torque: 1).
    pub fn width(&self) -> usize { output_width(self.raw().2) }

    pub(crate) fn raw(&self) -> DynamicsOutput {
        let (owner, id) = match self.owner { StateOwner::Link(id) => (0, id), StateOwner::Joint(id) => (1, id) };
        let family = match self.quantity {
            StateQuantity::Momentum => 0, StateQuantity::Force => 1, StateQuantity::Torque => 2,
            StateQuantity::SpatialMotion => 3, StateQuantity::Position => 4,
            StateQuantity::Rotation => 5, StateQuantity::Frame => 6,
        };
        (owner, id, family, self.derivative, self.frame == ReferenceFrame::World)
    }
}

impl RustCompiledRobot {
    pub(crate) fn check_dynamics_outputs(
        &self, outputs: &[DynamicsOutput], dynamics_order: usize,
    ) -> CoreResult<usize> {
        if dynamics_order == 0 {
            return Err(Error::new("selected dynamics requires dynamics_order >= 1"));
        }
        self.check_selected_outputs(outputs, dynamics_order + 2)
    }

    pub(crate) fn check_selected_outputs(&self, outputs: &[DynamicsOutput], order: usize) -> CoreResult<usize> {
        if order == 0 { return Err(Error::new("selected output order must be positive")); }
        let mut rows = 0;
        for &(owner, id, family, time, world) in outputs {
            let owners = match owner {
                0 => self.link_num,
                1 => self.joint_num,
                _ => return Err(Error::new("invalid dynamics output owner")),
            };
            let count = match family {
                0 => order.saturating_sub(1),
                1 | 2 => order.saturating_sub(2),
                3 => order.saturating_sub(1),
                4..=6 => 1,
                _ => return Err(Error::new("invalid dynamics output family")),
            };
            if id >= owners || time >= count || (family == 2 && (owner != 1 || world)) {
                return Err(Error::new("invalid dynamics output index, order or frame"));
            }
            rows += output_width(family);
        }
        Ok(rows)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dynamics_selected_tangent_into(
        &self, motion: &[f64], directions: &[f64], outputs: &[DynamicsOutput],
        dynamics_order: usize, gravity: [f64; 3],
        primal: &mut DynamicsCmtmWorkspace, tangent: &mut DynamicsCmtmTangentWorkspace,
        out: &mut [f64],
    ) {
        self.dynamics_cmtm_into(motion, dynamics_order, gravity, primal);
        self.selected_tangent_from_state_into(motion, directions, outputs, dynamics_order + 2, gravity, primal, tangent, out);
    }

    pub(crate) fn selected_tangent_from_state_into(
        &self, motion: &[f64], directions: &[f64], outputs: &[DynamicsOutput],
        kin_order: usize, gravity: [f64;3], primal: &mut DynamicsCmtmWorkspace,
        tangent: &mut DynamicsCmtmTangentWorkspace, out: &mut [f64],
    ) {
        let dynamics_order = kin_order.saturating_sub(2);
        if outputs.iter().any(|x| x.2 < 3) {
            self.dynamics_cmtm_link_tangent_from_state_into(motion, directions, dynamics_order, gravity, primal, tangent);
        } else {
            self.kinematics_cmtm_tangent_from_state_into(motion, directions, kin_order, &mut primal.cmtm, tangent);
        }
        let cols = tangent.rhs_cols;
        let mut row = 0;
        for &(owner, id, family, time, world) in outputs {
            if family >= 4 {
                let mat = mat4_from_flat(if owner == 0 { &primal.cmtm.link_mat } else { &primal.cmtm.joint_mat }, id);
                for col in 0..cols {
                    let dmat = crate::cmtm_generic::tangent_mat4(if owner == 0 { &tangent.link_mat } else { &tangent.joint_mat }, id, cols, col);
                    let value = pose_tangent(mat, dmat, family, world);
                    for c in 0..output_width(family) { out[(row+c)*cols+col] = value[c]; }
                }
                row += output_width(family);
                continue;
            }
            if family == 3 && !world {
                let derivatives = if owner == 0 { &tangent.link_vecs } else { &tangent.joint_vecs };
                for c in 0..6 { for col in 0..cols {
                    out[(row + c) * cols + col] = derivatives[((id * (kin_order - 1) + time) * 6 + c) * cols + col];
                }}
                row += 6;
                continue;
            }
            if family == 2 {
                for col in 0..cols {
                    out[row * cols + col] = tangent.joint_torque[(id * dynamics_order + time) * cols + col];
                }
                row += 1;
                continue;
            }
            let series_order = dynamics_order + if family == 0 || family == 3 { 1 } else { 0 };
            let (values, derivatives) = match (owner, family) {
                (0, 3) => (&primal.cmtm.link_vecs, &tangent.link_vecs),
                (1, 3) => (&primal.cmtm.joint_vecs, &tangent.joint_vecs),
                (0, 0) => (&primal.link_momentum, &tangent.link_momentum),
                (0, _) => (&primal.link_force, &tangent.link_force),
                (_, 0) => (&primal.joint_momentum, &tangent.joint_momentum),
                _ => (&primal.joint_force, &tangent.joint_force),
            };
            for col in 0..cols {
                if !world {
                    for c in 0..6 {
                        out[(row + c) * cols + col] = derivatives[((id * series_order + time) * 6 + c) * cols + col];
                    }
                    continue;
                }
                // A joint wrench is expressed in its child link frame.
                let link = if owner == 0 { id } else { self.child_link[id] };
                let order = time + 1;
                let vec_len = order.saturating_sub(1) * 6;
                let mat = mat4_from_flat(&primal.cmtm.link_mat, link);
                let dmat = crate::cmtm_generic::tangent_mat4(&tangent.link_mat, link, cols, col);
                let vecs = cmtm_vecs_slice(&primal.cmtm.link_vecs, link, kin_order);
                let dvecs = crate::cmtm_generic::tangent_cmtm_vecs(&tangent.link_vecs, link, kin_order, cols, col);
                let values = cmvec_slice(values, id, series_order);
                let values = if family == 3 { swap_spatial(values) } else { values.to_vec() };
                let derivatives = crate::cmtm_generic::tangent_cmvecs(derivatives, id, series_order, cols, col);
                let derivatives = if family == 3 { swap_spatial(&derivatives) } else { derivatives };
                let mut blocks = vec![[[0.0; 6]; 6]; order];
                let mut dblocks = vec![[[0.0; 6]; 6]; order];
                let mut value = vec![0.0; order * 6];
                let mut derivative = vec![0.0; order * 6];
                cmtm_apply_mat_adj_wrench_tangent_into(
                    mat, &vecs[..vec_len], dmat, &dvecs[..vec_len],
                    &values[..order * 6], &derivatives[..order * 6], order, &primal.cmtm.factorial,
                    &mut blocks, &mut dblocks, &mut value, &mut derivative,
                );
                for c in 0..6 { out[(row + c) * cols + col] = derivative[time * 6 + if family == 3 { (c+3)%6 } else { c }]; }
            }
            row += 6;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dynamics_selected_reverse_into(
        &self, motion: &[f64], rhs: &[f64], outputs: &[DynamicsOutput],
        dynamics_order: usize, gravity: [f64; 3], cols: usize,
        primal: &mut DynamicsCmtmWorkspace, out: &mut [f64],
    ) {
        self.dynamics_cmtm_into(motion, dynamics_order, gravity, primal);
        self.selected_reverse_from_state_into(motion, rhs, outputs, dynamics_order + 2, gravity, cols, primal, out);
    }

    pub(crate) fn selected_reverse_from_state_into(
        &self, motion: &[f64], rhs: &[f64], outputs: &[DynamicsOutput],
        kin_order: usize, gravity: [f64;3], cols: usize,
        primal: &mut DynamicsCmtmWorkspace, out: &mut [f64],
    ) {
        let dynamics_order = kin_order.saturating_sub(2);
        let has_dynamics = outputs.iter().any(|x| x.2 < 3);
        let momentum_order = dynamics_order + 1;
        let vec_len = (kin_order - 1) * 6;
        let mut lm = vec![0.0; if has_dynamics { self.link_num * momentum_order * 6 * cols } else { 0 }];
        let mut lf = vec![0.0; if has_dynamics { self.link_num * dynamics_order * 6 * cols } else { 0 }];
        let mut jm = vec![0.0; if has_dynamics { self.joint_num * momentum_order * 6 * cols } else { 0 }];
        let mut jf = vec![0.0; if has_dynamics { self.joint_num * dynamics_order * 6 * cols } else { 0 }];
        let mut jt = vec![0.0; if has_dynamics { self.joint_num * dynamics_order * cols } else { 0 }];
        let mut mat_bar = vec![0.0; self.link_num * 16 * cols];
        let mut vec_bar = vec![0.0; self.link_num * vec_len * cols];
        let mut joint_mat_bar = vec![0.0; self.joint_num * 16 * cols];
        let mut joint_vec_bar = vec![0.0; self.joint_num * vec_len * cols];
        let mut row = 0;
        for &(owner, id, family, time, world) in outputs {
            if family >= 4 {
                let mat = mat4_from_flat(if owner == 0 { &primal.cmtm.link_mat } else { &primal.cmtm.joint_mat }, id);
                let target = if owner == 0 { &mut mat_bar } else { &mut joint_mat_bar };
                // Transpose of the small pose projection (16 entries), never a motion Jacobian.
                for r in 0..3 { for c in 0..4 {
                    let mut basis = [[0.0;4];4]; basis[r][c] = 1.0;
                    let value = pose_tangent(mat, basis, family, world);
                    for col in 0..cols { for k in 0..output_width(family) {
                        target[(id*16+r*4+c)*cols+col] += value[k]*rhs[(row+k)*cols+col];
                    }}
                }}
                row += output_width(family);
                continue;
            }
            if family == 3 && !world {
                let target = if owner == 0 { &mut vec_bar } else { &mut joint_vec_bar };
                for c in 0..6 { for col in 0..cols {
                    target[((id * (kin_order - 1) + time) * 6 + c) * cols + col] += rhs[(row + c) * cols + col];
                }}
                row += 6;
                continue;
            }
            if family == 2 {
                for col in 0..cols {
                    jt[(id * dynamics_order + time) * cols + col] += rhs[row * cols + col];
                }
                row += 1;
                continue;
            }
            let series_order = dynamics_order + if family == 0 || family == 3 { 1 } else { 0 };
            let values = match (owner, family) {
                (0, 0) => &primal.link_momentum,
                (0, 3) => &primal.cmtm.link_vecs,
                (1, 3) => &primal.cmtm.joint_vecs,
                (0, _) => &primal.link_force,
                (_, 0) => &primal.joint_momentum,
                _ => &primal.joint_force,
            };
            if !world {
                let local_bar = match (owner, family) {
                    (0, 0) => &mut lm, (0, 3) => &mut vec_bar, (1, 3) => &mut joint_vec_bar,
                    (0, _) => &mut lf, (_, 0) => &mut jm, _ => &mut jf,
                };
                for c in 0..6 { for col in 0..cols {
                    local_bar[((id * series_order + time) * 6 + c) * cols + col] += rhs[(row + c) * cols + col];
                }}
                row += 6;
                continue;
            }
            let link = if owner == 0 { id } else { self.child_link[id] };
            let order = time + 1;
            let transport_vec_len = order.saturating_sub(1) * 6;
            let mat = mat4_from_flat(&primal.cmtm.link_mat, link);
            let vecs = cmtm_vecs_slice(&primal.cmtm.link_vecs, link, kin_order);
            let values = cmvec_slice(values, id, series_order);
            let values = if family == 3 { swap_spatial(values) } else { values.to_vec() };
            for col in 0..cols {
                let mut seed = vec![0.0; order * 6];
                for c in 0..6 { seed[time * 6 + if family == 3 { (c+3)%6 } else { c }] = rhs[(row + c) * cols + col]; }
                let mut local_seed = vec![0.0; order * 6];
                let mut vec_seed = vec![0.0; transport_vec_len];
                let mut mat_seed = [[0.0; 4]; 4];
                let mut a = vec![[[0.0; 3]; 3]; order];
                let mut c = vec![[[0.0; 3]; 3]; order];
                let mut ab = vec![[[0.0; 3]; 3]; order];
                let mut cb = vec![[[0.0; 3]; 3]; order];
                let mut scaled = vec![0.0; transport_vec_len];
                cmtm_accumulate_mat_adj_wrench_series_reverse_accumulate_into(
                    mat, &vecs[..transport_vec_len], &values[..order * 6], &seed,
                    order, &primal.cmtm.factorial, &mut scaled, &mut a, &mut c, &mut ab, &mut cb,
                    &mut local_seed, &mut vec_seed, &mut mat_seed,
                );
                let local_bar = match (owner, family) {
                    (0, 0) => &mut lm, (0, 3) => &mut vec_bar, (1, 3) => &mut joint_vec_bar,
                    (0, _) => &mut lf, (_, 0) => &mut jm, _ => &mut jf,
                };
                for i in 0..order * 6 {
                    local_bar[(id * series_order * 6 + i) * cols + col] += local_seed[if family == 3 { (i/6)*6+(i%6+3)%6 } else { i }];
                }
                for i in 0..transport_vec_len {
                    vec_bar[(link * vec_len + i) * cols + col] += vec_seed[i];
                }
                for r in 0..4 { for c in 0..4 {
                    mat_bar[(link * 16 + r * 4 + c) * cols + col] += mat_seed[r][c];
                }}
            }
            row += 6;
        }
        if has_dynamics {
            self.dynamics_cmtm_reverse_from_state_into(
                motion, &lm, &lf, &jm, &jf, &jt, dynamics_order, gravity, cols,
                None, Some((&mat_bar, &vec_bar)), Some((&joint_mat_bar, &joint_vec_bar)), primal, out,
            );
        } else {
            self.kinematics_cmtm_outward_reverse_from_state_into(
                motion, kin_order, &mat_bar, &vec_bar, &joint_mat_bar, &joint_vec_bar, cols, &mut primal.cmtm, out,
            );
        }
    }
}

fn output_width(family: usize) -> usize {
    match family { 2 => 1, 4 | 5 => 3, _ => 6 }
}

// Ad_motion(T) = P Ad_wrench(T) P, with P swapping angular/linear halves.
pub(crate) fn swap_spatial(values: &[f64]) -> Vec<f64> {
    (0..values.len()).map(|i| values[(i/6)*6+(i%6+3)%6]).collect()
}

fn pose_tangent(mat: [[f64;4];4], dmat: [[f64;4];4], family: usize, world: bool) -> [f64;6] {
    let body = mat4_mul(mat4_inv_se3(mat), dmat);
    let tangent = if world { mat4_mul(dmat, mat4_inv_se3(mat)) } else { body };
    let v = vee_se3(tangent);
    if family == 4 {
        if world { [dmat[0][3], dmat[1][3], dmat[2][3], 0.0,0.0,0.0] }
        else { [v[3],v[4],v[5],0.0,0.0,0.0] }
    } else { v }
}

impl RustCompiledRobot {
    /// Evaluate kinetic energy and its gradient with respect to the compact
    /// `(q, qdot)` CMTM input.  The gradient is obtained by seeding each
    /// link's local velocity with the energy derivative and using the shared
    /// kinematics reverse recurrence; no dynamics or basis tangent is built.
    pub(crate) fn kinetic_energy_gradient_into(
        &self, motion: &[f64], primal: &mut CmtmWorkspace, gradient: &mut [f64],
    ) -> f64 {
        const ORDER: usize = 2;
        self.kinematics_cmtm_into(motion, ORDER, primal);
        let mut energy = 0.0;
        let mut link_vec_bar = vec![0.0; self.link_num * 6];
        for link in 0..self.link_num {
            let velocity = vec6_from_flat(cmtm_vecs_slice(&primal.link_vecs, link, ORDER), 0);
            let iv = mat6_vec6(self.link_inertia[link], velocity);
            let itv = mat6_transpose_vec6(self.link_inertia[link], velocity);
            energy += 0.5 * velocity.iter().zip(iv.iter()).map(|(a, b)| a * b).sum::<f64>();
            for c in 0..6 {
                // `0.5 * v^T I v` differentiates to `0.5 * (I + I^T) v`.
                // Spatial inertias are symmetric, but retaining both terms
                // makes this correct for every accepted inertia matrix.
                link_vec_bar[link * 6 + c] = 0.5 * (iv[c] + itv[c]);
            }
        }
        let link_mat_zero = vec![0.0; self.link_num * 16];
        let joint_mat_zero = vec![0.0; self.joint_num * 16];
        let joint_vec_zero = vec![0.0; self.joint_num * 6];
        self.kinematics_cmtm_outward_reverse_into(
            motion, ORDER, &link_mat_zero, &link_vec_bar, &joint_mat_zero,
            &joint_vec_zero, 1, primal, gradient,
        );
        energy
    }
}

impl RustCompiledRobot {
    /// True reverse VJP for local link wrenches expressed in world frame.
    /// The world transport is reversed into local wrench and kinematics
    /// cotangents, then the shared local dynamics/kinematics reverse kernels
    /// complete the propagation to motion.
    pub(crate) fn world_link_dynamics_cmtm_reverse_vjp_into(
        &self, motion: &[f64], momentum_cotangent: &[f64], force_cotangent: &[f64],
        dynamics_order: usize, gravity: [f64; 3], rhs_cols: usize, out: &mut [f64],
    ) {
        let kin_order = dynamics_order + 2;
        let input_len = self.dof * kin_order;
        let momentum_order = dynamics_order + 1;
        let vec_len = (kin_order - 1) * 6;
        let mut primal = DynamicsCmtmWorkspace::new(self, dynamics_order);
        self.dynamics_cmtm_into(motion, dynamics_order, gravity, &mut primal);
        let mut local_momentum_bar = vec![0.0; self.link_num * momentum_order * 6 * rhs_cols];
        let mut local_force_bar = vec![0.0; self.link_num * dynamics_order * 6 * rhs_cols];
        let mut link_mat_bar = vec![0.0; self.link_num * 16 * rhs_cols];
        let mut link_vec_bar = vec![0.0; self.link_num * vec_len * rhs_cols];

        for link in 0..self.link_num {
            let mat = mat4_from_flat(&primal.cmtm.link_mat, link);
            let vecs = cmtm_vecs_slice(&primal.cmtm.link_vecs, link, kin_order);
            let momentum = cmvec_slice(&primal.link_momentum, link, momentum_order);
            let force = cmvec_slice(&primal.link_force, link, dynamics_order);
            for rhs in 0..rhs_cols {
                // The transport reverse handles every output coefficient at
                // once.  This replaces the former O(order^2) sequence of
                // prefix-series calls and preserves cross-coefficient terms.
                let mut reverse_transport = |raw_rhs: &[f64], target: &[f64], order: usize,
                                             local_bar: &mut [f64]| {
                    let mut target_bar = vec![0.0; order * 6];
                    for i in 0..order * 6 {
                        target_bar[i] = target[(link * order * 6 + i) * rhs_cols + rhs];
                    }
                    let mut rhs_bar = vec![0.0; order * 6];
                    let mut vec_bar = vec![0.0; order.saturating_sub(1) * 6];
                    let mut mat_bar = [[0.0; 4]; 4];
                    let mut a = vec![[[0.0; 3]; 3]; order];
                    let mut c_blocks = vec![[[0.0; 3]; 3]; order];
                    let mut a_bar = vec![[[0.0; 3]; 3]; order];
                    let mut c_bar = vec![[[0.0; 3]; 3]; order];
                    let mut scaled = vec![0.0; order.saturating_sub(1) * 6];
                    cmtm_accumulate_mat_adj_wrench_series_reverse_accumulate_into(
                        mat, &vecs[..order.saturating_sub(1) * 6], raw_rhs, &target_bar,
                        order, &primal.factorial, &mut scaled, &mut a, &mut c_blocks,
                        &mut a_bar, &mut c_bar, &mut rhs_bar, &mut vec_bar, &mut mat_bar,
                    );
                    for i in 0..order * 6 {
                        local_bar[(link * order * 6 + i) * rhs_cols + rhs] += rhs_bar[i];
                    }
                    for i in 0..order.saturating_sub(1) * 6 {
                        link_vec_bar[(link * vec_len + i) * rhs_cols + rhs] += vec_bar[i];
                    }
                    for row in 0..4 { for col in 0..4 {
                        link_mat_bar[(link * 16 + row * 4 + col) * rhs_cols + rhs] += mat_bar[row][col];
                    }}
                };
                reverse_transport(momentum, momentum_cotangent, momentum_order, &mut local_momentum_bar);
                reverse_transport(force, force_cotangent, dynamics_order, &mut local_force_bar);
            }
        }

        let mut dynamics_out = vec![0.0; input_len * rhs_cols];
        self.dynamics_cmtm_reverse_into(
            motion, &local_momentum_bar, &local_force_bar,
            &vec![0.0; self.joint_num * momentum_order * 6 * rhs_cols],
            &vec![0.0; self.joint_num * dynamics_order * 6 * rhs_cols],
            &vec![0.0; self.joint_num * dynamics_order * rhs_cols], dynamics_order,
            gravity, rhs_cols, None, &mut primal, &mut dynamics_out,
        );
        let mut kinematics_out = vec![0.0; input_len * rhs_cols];
        let mut kinematics = CmtmWorkspace::new(self, kin_order);
        self.kinematics_cmtm_outward_reverse_into(
            motion, kin_order, &link_mat_bar, &link_vec_bar,
            &vec![0.0; self.joint_num * 16 * rhs_cols],
            &vec![0.0; self.joint_num * vec_len * rhs_cols], rhs_cols,
            &mut kinematics, &mut kinematics_out,
        );
        for i in 0..out.len() { out[i] = dynamics_out[i] + kinematics_out[i]; }
    }

    pub(crate) fn world_joint_dynamics_cmtm_vjp_into(
        &self, motion: &[f64], momentum_cotangent: &[f64], force_cotangent: &[f64],
        dynamics_order: usize, gravity: [f64; 3], rhs_cols: usize, out: &mut [f64],
    ) {
        let mut link_momentum_cotangent = vec![0.0; self.link_num * (dynamics_order + 1) * 6 * rhs_cols];
        let mut link_force_cotangent = vec![0.0; self.link_num * dynamics_order * 6 * rhs_cols];
        for joint in 0..self.joint_num {
            for &link in &self.link_subtree_links[self.child_link[joint]] {
                for time in 0..=dynamics_order {
                    for component in 0..6 { for rhs in 0..rhs_cols {
                        let source = ((joint * (dynamics_order + 1) + time) * 6 + component) * rhs_cols + rhs;
                        let target = ((link * (dynamics_order + 1) + time) * 6 + component) * rhs_cols + rhs;
                        link_momentum_cotangent[target] += momentum_cotangent[source];
                    }}
                }
                for time in 0..dynamics_order {
                    for component in 0..6 { for rhs in 0..rhs_cols {
                        let source = ((joint * dynamics_order + time) * 6 + component) * rhs_cols + rhs;
                        let target = ((link * dynamics_order + time) * 6 + component) * rhs_cols + rhs;
                        link_force_cotangent[target] += force_cotangent[source];
                    }}
                }
            }
        }
        self.world_link_dynamics_cmtm_reverse_vjp_into(
            motion, &link_momentum_cotangent, &link_force_cotangent,
            dynamics_order, gravity, rhs_cols, out,
        );
    }
}
