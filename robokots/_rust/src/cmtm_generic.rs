use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::spatial::*;
use crate::types::RustCompiledRobot;
use crate::workspace::CmtmWorkspace;

impl RustCompiledRobot {
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
