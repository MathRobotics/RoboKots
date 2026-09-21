//! Selected dynamics, local/world spatial motion and pose share one primal and
//! derivative recurrence. Neither product kernel forms a dense motion Jacobian.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::spatial::*;
use crate::types::RustCompiledRobot;
use crate::workspace::{DynamicsCmtmTangentWorkspace, DynamicsCmtmWorkspace};

/// (owner: link=0/joint=1, owner index, momentum=0/force=1/torque=2/kinematics=3,
/// ordinary time derivative index, world frame).
/// Kinematics index zero denotes velocity; pose families 4/5/6 are pos/rot/frame.
/// Pose derivatives use body tangents by default and spatial tangents in world.
pub(crate) type DynamicsOutput = (usize, usize, usize, usize, bool);

impl RustCompiledRobot {
    pub(crate) fn check_dynamics_outputs(
        &self, outputs: &[DynamicsOutput], dynamics_order: usize,
    ) -> PyResult<usize> {
        if dynamics_order == 0 {
            return Err(PyValueError::new_err("selected dynamics requires dynamics_order >= 1"));
        }
        let mut rows = 0;
        for &(owner, id, family, time, world) in outputs {
            let owners = match owner {
                0 => self.link_num,
                1 => self.joint_num,
                _ => return Err(PyValueError::new_err("invalid dynamics output owner")),
            };
            let count = match family {
                0 => dynamics_order + 1,
                1 | 2 => dynamics_order,
                3 => dynamics_order + 1,
                4..=6 => 1,
                _ => return Err(PyValueError::new_err("invalid dynamics output family")),
            };
            if id >= owners || time >= count || (family == 2 && (owner != 1 || world)) {
                return Err(PyValueError::new_err("invalid dynamics output index, order or frame"));
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
        self.dynamics_cmtm_link_tangent_into(
            motion, directions, dynamics_order, gravity, primal, tangent,
        );
        let kin_order = dynamics_order + 2;
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
                    &values[..order * 6], &derivatives[..order * 6], order, &primal.factorial,
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
        let kin_order = dynamics_order + 2;
        let momentum_order = dynamics_order + 1;
        let vec_len = (kin_order - 1) * 6;
        let mut lm = vec![0.0; self.link_num * momentum_order * 6 * cols];
        let mut lf = vec![0.0; self.link_num * dynamics_order * 6 * cols];
        let mut jm = vec![0.0; self.joint_num * momentum_order * 6 * cols];
        let mut jf = vec![0.0; self.joint_num * dynamics_order * 6 * cols];
        let mut jt = vec![0.0; self.joint_num * dynamics_order * cols];
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
                    order, &primal.factorial, &mut scaled, &mut a, &mut c, &mut ab, &mut cb,
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
        self.dynamics_cmtm_reverse_from_state_into(
            motion, &lm, &lf, &jm, &jf, &jt, dynamics_order, gravity, cols,
            None, Some((&mat_bar, &vec_bar)), Some((&joint_mat_bar, &joint_vec_bar)), primal, out,
        );
    }
}

fn output_width(family: usize) -> usize {
    match family { 2 => 1, 4 | 5 => 3, _ => 6 }
}

// Ad_motion(T) = P Ad_wrench(T) P, with P swapping angular/linear halves.
fn swap_spatial(values: &[f64]) -> Vec<f64> {
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
