use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::spatial::*;
use crate::types::RustCompiledRobot;
use crate::workspace::{CmtmWorkspace, DynamicsCmtmTangentWorkspace};

fn mat4_t(a: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for r in 0..4 { for c in 0..4 { out[r][c] = a[c][r]; } }
    out
}

fn add_mat4_in_place(dst: &mut [[f64; 4]; 4], src: [[f64; 4]; 4]) {
    for r in 0..4 { for c in 0..4 { dst[r][c] += src[r][c]; } }
}

/// Frobenius adjoint of `hat_se3`.
fn hat_se3_adjoint(m: [[f64; 4]; 4]) -> [f64; 6] {
    [
        m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1],
        m[0][3], m[1][3], m[2][3],
    ]
}

/// Add the adjoint of `vee_se3` to a matrix cotangent.
fn vee_se3_adjoint_add(dst: &mut [[f64; 4]; 4], v: [f64; 6]) {
    dst[2][1] += 0.5 * v[0]; dst[1][2] -= 0.5 * v[0];
    dst[0][2] += 0.5 * v[1]; dst[2][0] -= 0.5 * v[1];
    dst[1][0] += 0.5 * v[2]; dst[0][1] -= 0.5 * v[2];
    dst[0][3] += v[3]; dst[1][3] += v[4]; dst[2][3] += v[5];
}

/// Add the adjoint of the SE(3) inverse, `inv([R,p])=[R^T,-R^Tp]`.
fn mat4_inv_se3_adjoint_add(dst: &mut [[f64; 4]; 4], e: [[f64; 4]; 4], g: [[f64; 4]; 4]) {
    let p = [e[0][3], e[1][3], e[2][3]];
    let gp = [g[0][3], g[1][3], g[2][3]];
    for r in 0..3 {
        for c in 0..3 { dst[r][c] += g[c][r] - p[r] * gp[c]; }
        dst[r][3] -= e[r][0] * gp[0] + e[r][1] * gp[1] + e[r][2] * gp[2];
    }
}

fn reverse_cmtm_recovery(
    blocks: &[[[f64; 4]; 4]], vecs: &[f64], order: usize, fact: &[f64],
    block_bar: &mut [[[f64; 4]; 4]], vec_bar: &mut [f64],
) {
    let e = blocks[0];
    let e_inv = mat4_inv_se3(e);
    let mut e_inv_bar = [[0.0; 4]; 4];
    for i in (0..order - 1).rev() {
        let mut s = scale_mat4(blocks[i + 1], (i + 1) as f64);
        for j in 0..i {
            s = sub_mat4(s, mat4_mul_hat_se3(blocks[i - j], scale6(vec6_from_flat(vecs, j), 1.0 / fact[j])));
        }
        let mut d_bar = [[0.0; 4]; 4];
        let u_bar = vec6_from_flat(vec_bar, i);
        vee_se3_adjoint_add(&mut d_bar, scale6(u_bar, fact[i]));
        add_mat4_in_place(&mut e_inv_bar, mat4_mul(d_bar, mat4_t(s)));
        let s_bar = mat4_mul(mat4_t(e_inv), d_bar);
        add_mat4_in_place(&mut block_bar[i + 1], scale_mat4(s_bar, (i + 1) as f64));
        for j in 0..i {
            let u = scale6(vec6_from_flat(vecs, j), 1.0 / fact[j]);
            let neg = scale_mat4(s_bar, -1.0);
            add_mat4_in_place(&mut block_bar[i - j], mat4_mul(neg, mat4_t(hat_se3(u))));
            let h_bar = mat4_mul(mat4_t(blocks[i - j]), neg);
            let v_bar = scale6(hat_se3_adjoint(h_bar), 1.0 / fact[j]);
            for c in 0..6 { vec_bar[j * 6 + c] += v_bar[c]; }
        }
    }
    mat4_inv_se3_adjoint_add(&mut block_bar[0], e, e_inv_bar);
}

fn reverse_cmtm_blocks(
    blocks: &[[[f64; 4]; 4]], vecs: &[f64], order: usize, fact: &[f64],
    block_bar: &mut [[[f64; 4]; 4]], vec_bar: &mut [f64],
) {
    for k in (1..order).rev() {
        let g = scale_mat4(block_bar[k], 1.0 / k as f64);
        for i in 0..k {
            let prev = k - i - 1;
            let v = scale6(vec6_from_flat(vecs, i), 1.0 / fact[i]);
            add_mat4_in_place(&mut block_bar[prev], mat4_mul(g, mat4_t(hat_se3(v))));
            let h_bar = mat4_mul(mat4_t(blocks[prev]), g);
            let v_bar = scale6(hat_se3_adjoint(h_bar), 1.0 / fact[i]);
            for c in 0..6 { vec_bar[i * 6 + c] += v_bar[c]; }
        }
    }
}

impl RustCompiledRobot {
    /// Seed the CMTM tangent recurrence from a block of motion directions.
    ///
    /// `motion_tangent` is scalar-major: `(dof * order, rhs_cols)`.  The
    /// resulting joint transform and joint-motion tangents are exact
    /// derivatives of the corresponding primal CMTM inputs.  Propagating
    /// these seeds through link transforms is deliberately kept in the
    /// subsequent CMTM-product tangent step.
    #[allow(dead_code)]
    pub(crate) fn cmtm_joint_tangent_seed_into(
        &self,
        motion: &[f64],
        motion_tangent: &[f64],
        order: usize,
        tangent: &mut DynamicsCmtmTangentWorkspace,
    ) {
        debug_assert_eq!(motion.len(), self.dof * order);
        debug_assert_eq!(motion_tangent.len(), self.dof * order * tangent.rhs_cols);
        tangent.clear();

        for j in 0..self.joint_num {
            let qi = self.q_index[j];
            if qi < 0 {
                continue;
            }
            let qi = qi as usize;
            let motion_start = qi * order;
            let rotation_derivative = rot_axis_derivative(self.axis[j], motion[motion_start]);
            for rhs_col in 0..tangent.rhs_cols {
                let dq = motion_tangent[motion_start * tangent.rhs_cols + rhs_col];
                let mat_start = (j * 16) * tangent.rhs_cols + rhs_col;
                for row in 0..3 {
                    for col in 0..3 {
                        tangent.joint_mat[mat_start + (row * 4 + col) * tangent.rhs_cols] =
                            rotation_derivative[row][col] * dq;
                    }
                }
                for time_order in 0..order - 1 {
                    let dvalue = motion_tangent
                        [(motion_start + time_order + 1) * tangent.rhs_cols + rhs_col];
                    let vec_start =
                        (j * (order - 1) * 6 + time_order * 6) * tangent.rhs_cols + rhs_col;
                    for axis_index in 0..3 {
                        tangent.joint_vecs[vec_start + axis_index * tangent.rhs_cols] =
                            self.axis[j][axis_index] * dvalue;
                    }
                }
            }
        }
    }

    /// Propagate analytic CMTM kinematics tangents for all RHS columns.
    ///
    /// This is the forward-mode kernel used by higher-order torque Jv.  The
    /// primal workspace is computed once, while each RHS column only carries
    /// the linearised CMTM recurrence.
    #[allow(dead_code)]
    pub(crate) fn kinematics_cmtm_tangent_into(
        &self,
        motion: &[f64],
        motion_tangent: &[f64],
        order: usize,
        primal: &mut CmtmWorkspace,
        tangent: &mut DynamicsCmtmTangentWorkspace,
    ) {
        debug_assert_eq!(motion.len(), self.dof * order);
        debug_assert_eq!(motion_tangent.len(), self.dof * order * tangent.rhs_cols);
        self.kinematics_cmtm_into(motion, order, primal);
        self.cmtm_joint_tangent_seed_into(motion, motion_tangent, order, tangent);

        for j in 0..self.joint_num {
            let parent = self.parent_link[j];
            let child = self.child_link[j];
            let parent_mat = mat4_from_flat(&primal.link_mat, parent);
            let parent_vecs = cmtm_vecs_slice(&primal.link_vecs, parent, order);
            let joint_mat = mat4_from_flat(&primal.joint_mat, j);
            let rel_mat = mat4_mul(
                mat4_from_rot_pos(self.origin_r[j], self.origin_p[j]),
                joint_mat,
            );
            let joint_vecs = cmtm_vecs_slice(&primal.joint_vecs, j, order);

            for rhs_col in 0..tangent.rhs_cols {
                let dl_mat = tangent_mat4(&tangent.link_mat, parent, tangent.rhs_cols, rhs_col);
                let dl_vecs = tangent_cmtm_vecs(
                    &tangent.link_vecs,
                    parent,
                    order,
                    tangent.rhs_cols,
                    rhs_col,
                );
                let d_joint_mat =
                    tangent_mat4(&tangent.joint_mat, j, tangent.rhs_cols, rhs_col);
                let dr_mat = mat4_mul(
                    mat4_from_rot_pos(self.origin_r[j], self.origin_p[j]),
                    d_joint_mat,
                );
                let dr_vecs = tangent_cmtm_vecs(
                    &tangent.joint_vecs,
                    j,
                    order,
                    tangent.rhs_cols,
                    rhs_col,
                );
                let mut l_blocks = vec![[[0.0; 4]; 4]; order];
                let mut r_blocks = vec![[[0.0; 4]; 4]; order];
                let mut out_blocks = vec![[[0.0; 4]; 4]; order];
                let mut hats = vec![[[0.0; 4]; 4]; order];
                let mut out_vecs = vec![0.0; (order - 1) * 6];
                let mut dl_blocks = vec![[[0.0; 4]; 4]; order];
                let mut dr_blocks = vec![[[0.0; 4]; 4]; order];
                let mut dout_blocks = vec![[[0.0; 4]; 4]; order];
                let mut dout_vecs = vec![0.0; (order - 1) * 6];
                let (_, dchild_mat) = cmtm_multiply_tangent_into(
                    parent_mat,
                    parent_vecs,
                    dl_mat,
                    &dl_vecs,
                    rel_mat,
                    joint_vecs,
                    dr_mat,
                    &dr_vecs,
                    order,
                    &primal.factorial,
                    &mut l_blocks,
                    &mut r_blocks,
                    &mut out_blocks,
                    &mut hats,
                    &mut out_vecs,
                    &mut dl_blocks,
                    &mut dr_blocks,
                    &mut dout_blocks,
                    &mut dout_vecs,
                );
                set_tangent_mat4(
                    &mut tangent.link_mat,
                    child,
                    tangent.rhs_cols,
                    rhs_col,
                    dchild_mat,
                );
                set_tangent_cmtm_vecs(
                    &mut tangent.link_vecs,
                    child,
                    order,
                    tangent.rhs_cols,
                    rhs_col,
                    &dout_vecs,
                );
            }
        }
    }

    /// Reverse CMTM kinematics for spatial-vector outputs without forming a
    /// basis of input tangents.  Cotangents use scalar-major `(value, rhs)`
    /// storage; this is the layout exposed by the Python VJP API.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn kinematics_cmtm_outward_reverse_into(
        &self,
        motion: &[f64], order: usize,
        link_mat_cotangent: &[f64], link_vec_cotangent: &[f64],
        joint_mat_cotangent: &[f64], joint_vec_cotangent: &[f64], rhs_cols: usize,
        primal: &mut CmtmWorkspace, motion_cotangent: &mut [f64],
    ) {
        self.kinematics_cmtm_into(motion, order, primal);
        motion_cotangent.fill(0.0);
        let vec_len = (order - 1) * 6;
        let mut link_mat_bar = vec![[[0.0; 4]; 4]; self.link_num];
        let mut link_vec_bar = vec![0.0; self.link_num * vec_len];
        let mut joint_vec_bar = vec![0.0; self.joint_num * vec_len];
        let mut joint_mat_bar = vec![[[0.0; 4]; 4]; self.joint_num];

        for rhs in 0..rhs_cols {
            link_mat_bar.fill([[0.0; 4]; 4]);
            link_vec_bar.fill(0.0);
            joint_vec_bar.fill(0.0);
            joint_mat_bar.fill([[0.0; 4]; 4]);
            for link in 0..self.link_num {
                for row in 0..4 { for col in 0..4 {
                    link_mat_bar[link][row][col] = link_mat_cotangent[(link * 16 + row * 4 + col) * rhs_cols + rhs];
                }}
                for i in 0..vec_len {
                    link_vec_bar[link * vec_len + i] = link_vec_cotangent[(link * vec_len + i) * rhs_cols + rhs];
                }
            }
            for joint in 0..self.joint_num {
                for row in 0..4 { for col in 0..4 {
                    joint_mat_bar[joint][row][col] = joint_mat_cotangent[(joint * 16 + row * 4 + col) * rhs_cols + rhs];
                }}
                for i in 0..vec_len {
                    joint_vec_bar[joint * vec_len + i] = joint_vec_cotangent[(joint * vec_len + i) * rhs_cols + rhs];
                }
            }

            for j in (0..self.joint_num).rev() {
                let parent = self.parent_link[j];
                let child = self.child_link[j];
                let parent_vec = cmtm_vecs_slice(&primal.link_vecs, parent, order);
                let child_vec = cmtm_vecs_slice(&primal.link_vecs, child, order);
                let joint_vec = cmtm_vecs_slice(&primal.joint_vecs, j, order);
                let parent_mat = mat4_from_flat(&primal.link_mat, parent);
                let child_mat = mat4_from_flat(&primal.link_mat, child);
                let origin = mat4_from_rot_pos(self.origin_r[j], self.origin_p[j]);
                let joint_mat = mat4_from_flat(&primal.joint_mat, j);
                let rel_mat = mat4_mul(origin, joint_mat);
                let mut left = vec![[[0.0; 4]; 4]; order];
                let mut right = vec![[[0.0; 4]; 4]; order];
                let mut child_blocks = vec![[[0.0; 4]; 4]; order];
                cmtm_mat_blocks_into(parent_mat, parent_vec, order, &primal.factorial, &mut left);
                cmtm_mat_blocks_into(rel_mat, joint_vec, order, &primal.factorial, &mut right);
                cmtm_mat_blocks_into(child_mat, child_vec, order, &primal.factorial, &mut child_blocks);
                let mut child_bar = vec![[[0.0; 4]; 4]; order];
                child_bar[0] = link_mat_bar[child];
                let mut child_vec_bar = link_vec_bar[child * vec_len..(child + 1) * vec_len].to_vec();
                reverse_cmtm_recovery(&child_blocks, child_vec, order, &primal.factorial, &mut child_bar, &mut child_vec_bar);

                let mut left_bar = vec![[[0.0; 4]; 4]; order];
                let mut right_bar = vec![[[0.0; 4]; 4]; order];
                for k in 0..order {
                    for i in 0..=k {
                        add_mat4_in_place(&mut left_bar[i], mat4_mul(child_bar[k], mat4_t(right[k - i])));
                        add_mat4_in_place(&mut right_bar[k - i], mat4_mul(mat4_t(left[i]), child_bar[k]));
                    }
                }
                let mut parent_vec_bar = vec![0.0; vec_len];
                let mut rel_vec_bar = joint_vec_bar[j * vec_len..(j + 1) * vec_len].to_vec();
                reverse_cmtm_blocks(&left, parent_vec, order, &primal.factorial, &mut left_bar, &mut parent_vec_bar);
                reverse_cmtm_blocks(&right, joint_vec, order, &primal.factorial, &mut right_bar, &mut rel_vec_bar);
                add_mat4_in_place(&mut link_mat_bar[parent], left_bar[0]);
                for i in 0..vec_len { link_vec_bar[parent * vec_len + i] += parent_vec_bar[i]; }
                let joint_from_rel = mat4_mul(mat4_t(origin), right_bar[0]);
                add_mat4_in_place(&mut joint_mat_bar[j], joint_from_rel);
                for i in 0..vec_len { joint_vec_bar[j * vec_len + i] = rel_vec_bar[i]; }
            }
            for j in 0..self.joint_num {
                let qi = self.q_index[j];
                if qi < 0 { continue; }
                let qi = qi as usize;
                let start = qi * order;
                let drot = rot_axis_derivative(self.axis[j], motion[start]);
                let mut dq = 0.0;
                for r in 0..3 { for c in 0..3 { dq += joint_mat_bar[j][r][c] * drot[r][c]; } }
                motion_cotangent[start * rhs_cols + rhs] += dq;
                for time in 0..order - 1 {
                    let mut dv = 0.0;
                    for c in 0..3 { dv += self.axis[j][c] * joint_vec_bar[(j * vec_len + time * 6) + c]; }
                    motion_cotangent[(start + time + 1) * rhs_cols + rhs] += dv;
                }
            }
        }
    }

    pub(crate) fn check_cmtm_motion(&self, motion: &[f64], order: usize) -> PyResult<()> {
        if motion.len() != self.dof * order {
            return Err(PyValueError::new_err(
                "motion length must match robot dof * order",
            ));
        }
        Ok(())
    }

    pub(crate) fn kinematics_cmtm_into(
        &self,
        motion: &[f64],
        order: usize,
        ws: &mut CmtmWorkspace,
    ) {
        if order == 3 {
            self.kinematics_cmtm_order3_fast_into(motion, ws, true);
            return;
        }

        ws.clear();
        fill_factorial_table(&mut ws.factorial);
        set_mat4(&mut ws.link_mat, 0, eye4());

        for j in 0..self.joint_num {
            let parent = self.parent_link[j];
            let child = self.child_link[j];
            let parent_mat = mat4_from_flat(&ws.link_mat, parent);
            let parent_vecs = cmtm_vecs_slice(&ws.link_vecs, parent, order);
            let origin_mat = mat4_from_rot_pos(self.origin_r[j], self.origin_p[j]);
            let qi = self.q_index[j];

            ws.tmp_rel_vecs.fill(0.0);
            let (local_mat, rel_mat) = if qi >= 0 {
                let qi = qi as usize;
                let motion_start = qi * order;
                let local_mat =
                    mat4_from_rot_pos(rot_axis(self.axis[j], motion[motion_start]), [0.0; 3]);
                let rel_mat = mat4_mul(origin_mat, local_mat);
                for k in 0..order - 1 {
                    let value = motion[motion_start + k + 1];
                    ws.tmp_rel_vecs[k * 6] = self.axis[j][0] * value;
                    ws.tmp_rel_vecs[k * 6 + 1] = self.axis[j][1] * value;
                    ws.tmp_rel_vecs[k * 6 + 2] = self.axis[j][2] * value;
                }
                (local_mat, rel_mat)
            } else {
                (eye4(), origin_mat)
            };

            set_mat4(&mut ws.joint_mat, j, local_mat);
            set_cmtm_vecs_flat(&mut ws.joint_vecs, j, order, &ws.tmp_rel_vecs);
            let child_mat = cmtm_multiply_into(
                parent_mat,
                parent_vecs,
                rel_mat,
                &ws.tmp_rel_vecs,
                order,
                &ws.factorial,
                &mut ws.tmp_mat4_blocks_a,
                &mut ws.tmp_mat4_blocks_b,
                &mut ws.tmp_mat4_blocks_out,
                &mut ws.tmp_hat4_blocks,
                &mut ws.tmp_out_vecs,
            );
            set_mat4(&mut ws.link_mat, child, child_mat);
            set_cmtm_vecs_flat(&mut ws.link_vecs, child, order, &ws.tmp_out_vecs);
        }
    }

    pub(crate) fn kinematics_cmtm_order3_fast_into(
        &self,
        motion: &[f64],
        ws: &mut CmtmWorkspace,
        store_joints: bool,
    ) {
        if store_joints {
            ws.clear();
            set_eye3(&mut ws.fast_r, 0);
            set_mat4(&mut ws.link_mat, 0, eye4());
        } else {
            set_mat4(&mut ws.link_mat, 0, eye4());
            set_eye3(&mut ws.fast_r, 0);
            ws.fast_p[..3].fill(0.0);
            ws.fast_w[..3].fill(0.0);
            ws.fast_lin_v[..3].fill(0.0);
            ws.fast_alpha[..3].fill(0.0);
            ws.fast_lin_a[..3].fill(0.0);
            ws.link_vecs[..12].fill(0.0);
        }

        for j in 0..self.joint_num {
            let parent = self.parent_link[j];
            let child = self.child_link[j];
            let parent_r = mat3_from_flat(&ws.fast_r, parent);
            let parent_p = flat3(&ws.fast_p, parent);
            let parent_w = flat3(&ws.fast_w, parent);
            let parent_lin_v = flat3(&ws.fast_lin_v, parent);
            let parent_alpha = flat3(&ws.fast_alpha, parent);
            let parent_lin_a = flat3(&ws.fast_lin_a, parent);
            let joint_r0 = mat3_mul(parent_r, self.origin_r[j]);
            let joint_p = add3(parent_p, mat3_vec(parent_r, self.origin_p[j]));
            let rel = sub3(joint_p, parent_p);
            let qi = self.q_index[j];

            if qi >= 0 {
                let qi = qi as usize;
                let motion_start = qi * 3;
                let q = motion[motion_start];
                let v = motion[motion_start + 1];
                let a = motion[motion_start + 2];
                let axis_world = mat3_vec(joint_r0, self.axis[j]);
                let rj = rot_axis(self.axis[j], q);
                let child_r = mat3_mul(joint_r0, rj);
                let child_w = add3(parent_w, scale3(axis_world, v));
                let child_lin_v = add3(parent_lin_v, cross(parent_w, rel));
                let child_alpha = add3(
                    add3(parent_alpha, scale3(axis_world, a)),
                    cross(parent_w, scale3(axis_world, v)),
                );
                let child_lin_a = add3(
                    add3(parent_lin_a, cross(parent_alpha, rel)),
                    cross(parent_w, cross(parent_w, rel)),
                );
                let child_rt = mat3_transpose(child_r);
                let local_w = mat3_vec(child_rt, child_w);
                let local_lin_v = mat3_vec(child_rt, child_lin_v);
                let local_alpha = mat3_vec(child_rt, child_alpha);
                let local_lin_a =
                    sub3(mat3_vec(child_rt, child_lin_a), cross(local_w, local_lin_v));

                set_mat3(&mut ws.fast_r, child, child_r);
                set_flat3(&mut ws.fast_p, child, joint_p);
                set_flat3(&mut ws.fast_w, child, child_w);
                set_flat3(&mut ws.fast_lin_v, child, child_lin_v);
                set_flat3(&mut ws.fast_alpha, child, child_alpha);
                set_flat3(&mut ws.fast_lin_a, child, child_lin_a);
                set_mat4(&mut ws.link_mat, child, mat4_from_rot_pos(child_r, joint_p));
                set_vec6_flat(
                    &mut ws.link_vecs,
                    child * 2,
                    [
                        local_w[0],
                        local_w[1],
                        local_w[2],
                        local_lin_v[0],
                        local_lin_v[1],
                        local_lin_v[2],
                    ],
                );
                set_vec6_flat(
                    &mut ws.link_vecs,
                    child * 2 + 1,
                    [
                        local_alpha[0],
                        local_alpha[1],
                        local_alpha[2],
                        local_lin_a[0],
                        local_lin_a[1],
                        local_lin_a[2],
                    ],
                );
                if store_joints {
                    set_mat4(&mut ws.joint_mat, j, mat4_from_rot_pos(rj, [0.0; 3]));
                    set_vec6_flat(
                        &mut ws.joint_vecs,
                        j * 2,
                        [
                            self.axis[j][0] * v,
                            self.axis[j][1] * v,
                            self.axis[j][2] * v,
                            0.0,
                            0.0,
                            0.0,
                        ],
                    );
                    set_vec6_flat(
                        &mut ws.joint_vecs,
                        j * 2 + 1,
                        [
                            self.axis[j][0] * a,
                            self.axis[j][1] * a,
                            self.axis[j][2] * a,
                            0.0,
                            0.0,
                            0.0,
                        ],
                    );
                }
            } else {
                let child_r = joint_r0;
                let child_w = parent_w;
                let child_lin_v = add3(parent_lin_v, cross(parent_w, rel));
                let child_alpha = parent_alpha;
                let child_lin_a = add3(
                    add3(parent_lin_a, cross(parent_alpha, rel)),
                    cross(parent_w, cross(parent_w, rel)),
                );
                let child_rt = mat3_transpose(child_r);
                let local_w = mat3_vec(child_rt, child_w);
                let local_lin_v = mat3_vec(child_rt, child_lin_v);
                let local_alpha = mat3_vec(child_rt, child_alpha);
                let local_lin_a =
                    sub3(mat3_vec(child_rt, child_lin_a), cross(local_w, local_lin_v));

                set_mat3(&mut ws.fast_r, child, child_r);
                set_flat3(&mut ws.fast_p, child, joint_p);
                set_flat3(&mut ws.fast_w, child, child_w);
                set_flat3(&mut ws.fast_lin_v, child, child_lin_v);
                set_flat3(&mut ws.fast_alpha, child, child_alpha);
                set_flat3(&mut ws.fast_lin_a, child, child_lin_a);
                set_mat4(&mut ws.link_mat, child, mat4_from_rot_pos(child_r, joint_p));
                set_vec6_flat(
                    &mut ws.link_vecs,
                    child * 2,
                    [
                        local_w[0],
                        local_w[1],
                        local_w[2],
                        local_lin_v[0],
                        local_lin_v[1],
                        local_lin_v[2],
                    ],
                );
                set_vec6_flat(
                    &mut ws.link_vecs,
                    child * 2 + 1,
                    [
                        local_alpha[0],
                        local_alpha[1],
                        local_alpha[2],
                        local_lin_a[0],
                        local_lin_a[1],
                        local_lin_a[2],
                    ],
                );
                if store_joints {
                    set_mat4(&mut ws.joint_mat, j, eye4());
                }
            }
        }
    }
}

pub(crate) fn tangent_mat4(flat: &[f64], index: usize, rhs_cols: usize, rhs_col: usize) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            out[row][col] = flat[(index * 16 + row * 4 + col) * rhs_cols + rhs_col];
        }
    }
    out
}

fn set_tangent_mat4(
    flat: &mut [f64], index: usize, rhs_cols: usize, rhs_col: usize, mat: [[f64; 4]; 4],
) {
    for row in 0..4 {
        for col in 0..4 {
            flat[(index * 16 + row * 4 + col) * rhs_cols + rhs_col] = mat[row][col];
        }
    }
}

pub(crate) fn tangent_cmtm_vecs(
    flat: &[f64], index: usize, order: usize, rhs_cols: usize, rhs_col: usize,
) -> Vec<f64> {
    let scalar_start = index * (order - 1) * 6;
    (0..(order - 1) * 6)
        .map(|offset| flat[(scalar_start + offset) * rhs_cols + rhs_col])
        .collect()
}

pub(crate) fn set_tangent_cmtm_vecs(
    flat: &mut [f64], index: usize, order: usize, rhs_cols: usize, rhs_col: usize, vecs: &[f64],
) {
    let scalar_start = index * (order - 1) * 6;
    for (offset, value) in vecs.iter().enumerate() {
        flat[(scalar_start + offset) * rhs_cols + rhs_col] = *value;
    }
}

/// Access a generic CMTM vector series with `count` six-vectors.  Unlike
/// kinematics CMTM vectors, dynamics momentum/force series do not use the
/// `order - 1` convention.
pub(crate) fn set_tangent_cmvecs(
    flat: &mut [f64], index: usize, count: usize, rhs_cols: usize, rhs_col: usize, vecs: &[f64],
) {
    let scalar_start = index * count * 6;
    for (offset, value) in vecs.iter().enumerate() {
        flat[(scalar_start + offset) * rhs_cols + rhs_col] = *value;
    }
}

pub(crate) fn tangent_cmvecs(
    flat: &[f64], index: usize, count: usize, rhs_cols: usize, rhs_col: usize,
) -> Vec<f64> {
    let scalar_start = index * count * 6;
    (0..count * 6)
        .map(|offset| flat[(scalar_start + offset) * rhs_cols + rhs_col])
        .collect()
}
