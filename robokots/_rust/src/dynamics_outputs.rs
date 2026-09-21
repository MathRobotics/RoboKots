//! Selected dynamics and local spatial kinematics share one primal and
//! derivative recurrence. Neither product kernel forms a dense motion Jacobian.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::spatial::*;
use crate::types::RustCompiledRobot;
use crate::workspace::{DynamicsCmtmTangentWorkspace, DynamicsCmtmWorkspace};

/// (owner: link=0/joint=1, owner index, momentum=0/force=1/torque=2/kinematics=3,
/// ordinary time derivative index, world frame).
/// Kinematics index zero denotes velocity. Kinematic world/pose outputs are
/// not supported by this selector; dynamics keeps its existing world support.
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
                _ => return Err(PyValueError::new_err("invalid dynamics output family")),
            };
            if id >= owners || time >= count || (family == 2 && (owner != 1 || world))
                || (family == 3 && world) {
                return Err(PyValueError::new_err("invalid dynamics output index, order or frame"));
            }
            rows += if family == 2 { 1 } else { 6 };
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
            if family == 3 {
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
            let series_order = dynamics_order + if family == 0 { 1 } else { 0 };
            let (values, derivatives) = match (owner, family) {
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
                let derivatives = crate::cmtm_generic::tangent_cmvecs(derivatives, id, series_order, cols, col);
                let mut blocks = vec![[[0.0; 6]; 6]; order];
                let mut dblocks = vec![[[0.0; 6]; 6]; order];
                let mut value = vec![0.0; order * 6];
                let mut derivative = vec![0.0; order * 6];
                cmtm_apply_mat_adj_wrench_tangent_into(
                    mat, &vecs[..vec_len], dmat, &dvecs[..vec_len],
                    &values[..order * 6], &derivatives[..order * 6], order, &primal.factorial,
                    &mut blocks, &mut dblocks, &mut value, &mut derivative,
                );
                for c in 0..6 { out[(row + c) * cols + col] = derivative[time * 6 + c]; }
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
        let mut joint_vec_bar = vec![0.0; self.joint_num * vec_len * cols];
        let mut row = 0;
        for &(owner, id, family, time, world) in outputs {
            if family == 3 {
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
            let series_order = dynamics_order + if family == 0 { 1 } else { 0 };
            let (values, local_bar) = match (owner, family) {
                (0, 0) => (&primal.link_momentum, &mut lm),
                (0, _) => (&primal.link_force, &mut lf),
                (_, 0) => (&primal.joint_momentum, &mut jm),
                _ => (&primal.joint_force, &mut jf),
            };
            if !world {
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
            for col in 0..cols {
                let mut seed = vec![0.0; order * 6];
                for c in 0..6 { seed[time * 6 + c] = rhs[(row + c) * cols + col]; }
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
                for i in 0..order * 6 {
                    local_bar[(id * series_order * 6 + i) * cols + col] += local_seed[i];
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
            None, Some((&mat_bar, &vec_bar)), Some(&joint_vec_bar), primal, out,
        );
    }
}
