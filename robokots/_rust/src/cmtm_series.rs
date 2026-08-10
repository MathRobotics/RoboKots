use crate::spatial::*;
use crate::types::RustCompiledRobot;
use crate::workspace::DynamicsCmtmWorkspace;

fn gravity_is_zero(gravity: [f64; 3]) -> bool {
    gravity == [0.0; 3]
}

fn binomial(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    let mut value = 1.0;
    for i in 0..k {
        value *= (n - i) as f64 / (i + 1) as f64;
    }
    value
}

fn gravity_force_series_into(
    link_mat: [[f64; 4]; 4],
    link_vel: &[f64],
    inertia: [[f64; 6]; 6],
    gravity: [f64; 3],
    order: usize,
    local_gravity: &mut [f64],
    force: &mut [f64],
) {
    if order == 0 {
        return;
    }
    let rotation_t = mat3_transpose(mat3_from_mat4(link_mat));
    let gravity0 = mat3_vec(rotation_t, gravity);
    local_gravity[..3].copy_from_slice(&gravity0);

    // For a world-fixed vector expressed in a moving link frame,
    // g_L^(n+1) = -sum_k C(n,k) omega^(k) x g_L^(n-k).
    for n in 1..order {
        let mut derivative = [0.0; 3];
        for k in 0..n {
            let omega = vec6_from_flat(link_vel, k);
            let gravity_derivative = [
                local_gravity[(n - 1 - k) * 3],
                local_gravity[(n - 1 - k) * 3 + 1],
                local_gravity[(n - 1 - k) * 3 + 2],
            ];
            let term = cross([omega[0], omega[1], omega[2]], gravity_derivative);
            let coefficient = binomial(n - 1, k);
            for i in 0..3 {
                derivative[i] -= coefficient * term[i];
            }
        }
        local_gravity[n * 3..(n + 1) * 3].copy_from_slice(&derivative);
    }

    for n in 0..order {
        let acceleration = [
            0.0,
            0.0,
            0.0,
            local_gravity[n * 3],
            local_gravity[n * 3 + 1],
            local_gravity[n * 3 + 2],
        ];
        let gravity_force = mat6_vec6(inertia, acceleration);
        for i in 0..6 {
            force[n * 6 + i] = -gravity_force[i];
        }
    }
}

impl RustCompiledRobot {
    pub(crate) fn dynamics_cmtm_into(
        &self,
        motion: &[f64],
        dynamics_order: usize,
        gravity: [f64; 3],
        ws: &mut DynamicsCmtmWorkspace,
    ) {
        if dynamics_order == 1 && gravity_is_zero(gravity) {
            self.dynamics_cmtm_order1_full_into(motion, ws);
            return;
        }

        let kin_order = dynamics_order + 2;
        let momentum_order = dynamics_order + 1;
        ws.clear();
        fill_factorial_table(&mut ws.factorial);
        self.kinematics_cmtm_into(motion, kin_order, &mut ws.cmtm);

        for j in (0..self.joint_num).rev() {
            let child = self.child_link[j];
            let child_joint_ids = &self.link_child_joints[child];
            let link_vel = cmtm_vecs_slice(&ws.cmtm.link_vecs, child, kin_order);
            momentum_from_velocity_into(
                self.link_inertia[child],
                link_vel,
                momentum_order,
                &mut ws.tmp_link_momentum,
            );
            set_cmvec_flat(
                &mut ws.link_momentum,
                child,
                momentum_order,
                &ws.tmp_link_momentum,
            );

            if dynamics_order > 0 {
                force_from_velocity_momentum_into(
                    link_vel,
                    &ws.tmp_link_momentum,
                    dynamics_order,
                    &ws.factorial,
                    &mut ws.tmp_force,
                );
                if !gravity_is_zero(gravity) {
                    gravity_force_series_into(
                        mat4_from_flat(&ws.cmtm.link_mat, child),
                        link_vel,
                        self.link_inertia[child],
                        gravity,
                        dynamics_order,
                        &mut ws.tmp_local_gravity,
                        &mut ws.tmp_gravity_force,
                    );
                    for i in 0..dynamics_order * 6 {
                        ws.tmp_force[i] += ws.tmp_gravity_force[i];
                    }
                }
                set_cmvec_flat(&mut ws.link_force, child, dynamics_order, &ws.tmp_force);
            }

            ws.tmp_joint_momentum[..momentum_order * 6]
                .copy_from_slice(&ws.tmp_link_momentum[..momentum_order * 6]);
            for &child_joint_id in child_joint_ids {
                let child_joint_momentum =
                    cmvec_slice(&ws.joint_momentum, child_joint_id, momentum_order);
                let joint_r = mat3_from_mat4(mat4_from_flat(&ws.cmtm.joint_mat, child_joint_id));
                let rel_mat = mat4_from_rot_pos(
                    mat3_mul(self.origin_r[child_joint_id], joint_r),
                    self.origin_p[child_joint_id],
                );
                let rel_vecs = cmtm_vecs_slice(&ws.cmtm.joint_vecs, child_joint_id, kin_order);
                cmtm_accumulate_mat_adj_wrench_series_into(
                    rel_mat,
                    &rel_vecs[..(momentum_order - 1) * 6],
                    child_joint_momentum,
                    momentum_order,
                    &ws.factorial,
                    &mut ws.tmp_scaled_vecs,
                    &mut ws.tmp_wrench_adj_a_blocks,
                    &mut ws.tmp_wrench_adj_c_blocks,
                    &mut ws.tmp_joint_momentum,
                );
                if !gravity_is_zero(gravity) {
                    let child_gravity =
                        cmvec_slice(&ws.joint_gravity_force, child_joint_id, dynamics_order);
                    cmtm_accumulate_mat_adj_wrench_series_into(
                        rel_mat,
                        &rel_vecs[..dynamics_order.saturating_sub(1) * 6],
                        child_gravity,
                        dynamics_order,
                        &ws.factorial,
                        &mut ws.tmp_scaled_vecs,
                        &mut ws.tmp_wrench_adj_a_blocks,
                        &mut ws.tmp_wrench_adj_c_blocks,
                        &mut ws.tmp_gravity_force,
                    );
                }
            }

            set_cmvec_flat(
                &mut ws.joint_momentum,
                j,
                momentum_order,
                &ws.tmp_joint_momentum,
            );

            if dynamics_order > 0 {
                if !gravity_is_zero(gravity) {
                    set_cmvec_flat(
                        &mut ws.joint_gravity_force,
                        j,
                        dynamics_order,
                        &ws.tmp_gravity_force,
                    );
                }
                force_from_velocity_momentum_into(
                    link_vel,
                    &ws.tmp_joint_momentum,
                    dynamics_order,
                    &ws.factorial,
                    &mut ws.tmp_force,
                );
                if !gravity_is_zero(gravity) {
                    for i in 0..dynamics_order * 6 {
                        ws.tmp_force[i] += ws.tmp_gravity_force[i];
                    }
                }
                set_cmvec_flat(&mut ws.joint_force, j, dynamics_order, &ws.tmp_force);
                if self.q_index[j] >= 0 {
                    for k in 0..dynamics_order {
                        let f = vec6_from_flat(&ws.tmp_force, k);
                        ws.joint_torque[j * dynamics_order + k] =
                            dot3(self.axis[j], [f[0], f[1], f[2]]);
                    }
                }
            }
        }

        let world = 0usize;
        let world_vel = cmtm_vecs_slice(&ws.cmtm.link_vecs, world, kin_order);
        momentum_from_velocity_into(
            self.link_inertia[world],
            world_vel,
            momentum_order,
            &mut ws.tmp_link_momentum,
        );
        set_cmvec_flat(
            &mut ws.link_momentum,
            world,
            momentum_order,
            &ws.tmp_link_momentum,
        );
        if dynamics_order > 0 {
            force_from_velocity_momentum_into(
                world_vel,
                &ws.tmp_link_momentum,
                dynamics_order,
                &ws.factorial,
                &mut ws.tmp_force,
            );
            if !gravity_is_zero(gravity) {
                gravity_force_series_into(
                    mat4_from_flat(&ws.cmtm.link_mat, world),
                    world_vel,
                    self.link_inertia[world],
                    gravity,
                    dynamics_order,
                    &mut ws.tmp_local_gravity,
                    &mut ws.tmp_gravity_force,
                );
                for i in 0..dynamics_order * 6 {
                    ws.tmp_force[i] += ws.tmp_gravity_force[i];
                }
            }
            set_cmvec_flat(&mut ws.link_force, world, dynamics_order, &ws.tmp_force);
        }
    }

    pub(crate) fn dynamics_cmtm_order1_full_into(
        &self,
        motion: &[f64],
        ws: &mut DynamicsCmtmWorkspace,
    ) {
        ws.clear();
        self.kinematics_cmtm_into(motion, 3, &mut ws.cmtm);

        for j in (0..self.joint_num).rev() {
            let child = self.child_link[j];
            let child_joint_ids = &self.link_child_joints[child];
            let link_vel = cmtm_vecs_slice(&ws.cmtm.link_vecs, child, 3);
            let v0 = vec6_from_flat(link_vel, 0);
            let v1 = vec6_from_flat(link_vel, 1);
            let m0 = mat6_vec6(self.link_inertia[child], v0);
            let m1 = mat6_vec6(self.link_inertia[child], v1);
            set_vec6_flat(&mut ws.tmp_link_momentum, 0, m0);
            set_vec6_flat(&mut ws.tmp_link_momentum, 1, m1);
            set_cmvec_flat(&mut ws.link_momentum, child, 2, &ws.tmp_link_momentum);

            let link_force = add6(m1, hat_adj_wrench_vec6(v0, m0));
            set_vec6_flat(&mut ws.link_force, child, link_force);

            ws.tmp_joint_momentum[..12].copy_from_slice(&ws.tmp_link_momentum[..12]);
            for &child_joint_id in child_joint_ids {
                let child_joint_momentum = cmvec_slice(&ws.joint_momentum, child_joint_id, 2);
                let joint_r = mat3_from_mat4(mat4_from_flat(&ws.cmtm.joint_mat, child_joint_id));
                let rel_mat = mat4_from_rot_pos(
                    mat3_mul(self.origin_r[child_joint_id], joint_r),
                    self.origin_p[child_joint_id],
                );
                let rel_v =
                    vec6_from_flat(cmtm_vecs_slice(&ws.cmtm.joint_vecs, child_joint_id, 3), 0);
                let child_m0 = vec6_from_flat(child_joint_momentum, 0);
                let child_m1 = vec6_from_flat(child_joint_momentum, 1);
                let transported0 = mat_adj_wrench_vec6_from_mat4(rel_mat, child_m0);
                let transported1 = mat_adj_wrench_vec6_from_mat4(
                    rel_mat,
                    add6(child_m1, hat_adj_wrench_vec6(rel_v, child_m0)),
                );
                for i in 0..6 {
                    ws.tmp_joint_momentum[i] += transported0[i];
                    ws.tmp_joint_momentum[6 + i] += transported1[i];
                }
            }

            set_cmvec_flat(&mut ws.joint_momentum, j, 2, &ws.tmp_joint_momentum);

            let joint_m0 = vec6_from_flat(&ws.tmp_joint_momentum, 0);
            let joint_m1 = vec6_from_flat(&ws.tmp_joint_momentum, 1);
            let joint_force = add6(joint_m1, hat_adj_wrench_vec6(v0, joint_m0));
            set_vec6_flat(&mut ws.joint_force, j, joint_force);
            if self.q_index[j] >= 0 {
                ws.joint_torque[j] = dot3(
                    self.axis[j],
                    [joint_force[0], joint_force[1], joint_force[2]],
                );
            }
        }
    }

    pub(crate) fn dynamics_cmtm_order1_cached_into(
        &self,
        motion: &[f64],
        ws: &mut DynamicsCmtmWorkspace,
    ) {
        ws.cached_motion[..motion.len()].copy_from_slice(motion);
        self.kinematics_cmtm_order3_fast_into(motion, &mut ws.cmtm, false);

        for j in (0..self.joint_num).rev() {
            let child = self.child_link[j];
            let child_joint_ids = &self.link_child_joints[child];
            let link_vel = cmtm_vecs_slice(&ws.cmtm.link_vecs, child, 3);
            let v0 = vec6_from_flat(link_vel, 0);
            let v1 = vec6_from_flat(link_vel, 1);
            let m0 = mat6_vec6(self.link_inertia[child], v0);
            let m1 = mat6_vec6(self.link_inertia[child], v1);
            set_vec6_flat(&mut ws.tmp_link_momentum, 0, m0);
            set_vec6_flat(&mut ws.tmp_link_momentum, 1, m1);

            ws.tmp_joint_momentum[..12].copy_from_slice(&ws.tmp_link_momentum[..12]);
            for &child_joint_id in child_joint_ids {
                let child_joint_momentum = cmvec_slice(&ws.joint_momentum, child_joint_id, 2);
                let q_index = self.q_index[child_joint_id];
                let joint_r = if q_index >= 0 {
                    let q = motion[q_index as usize * 3];
                    rot_axis(self.axis[child_joint_id], q)
                } else {
                    eye3()
                };
                let rel_mat = mat4_from_rot_pos(
                    mat3_mul(self.origin_r[child_joint_id], joint_r),
                    self.origin_p[child_joint_id],
                );
                let rel_v = if q_index >= 0 {
                    let v = motion[q_index as usize * 3 + 1];
                    [
                        self.axis[child_joint_id][0] * v,
                        self.axis[child_joint_id][1] * v,
                        self.axis[child_joint_id][2] * v,
                        0.0,
                        0.0,
                        0.0,
                    ]
                } else {
                    [0.0; 6]
                };
                let child_m0 = vec6_from_flat(child_joint_momentum, 0);
                let child_m1 = vec6_from_flat(child_joint_momentum, 1);
                let transported0 = mat_adj_wrench_vec6_from_mat4(rel_mat, child_m0);
                let transported1 = mat_adj_wrench_vec6_from_mat4(
                    rel_mat,
                    add6(child_m1, hat_adj_wrench_vec6(rel_v, child_m0)),
                );
                for i in 0..6 {
                    ws.tmp_joint_momentum[i] += transported0[i];
                    ws.tmp_joint_momentum[6 + i] += transported1[i];
                }
            }

            set_cmvec_flat(&mut ws.joint_momentum, j, 2, &ws.tmp_joint_momentum);

            if self.q_index[j] >= 0 {
                let joint_m0 = vec6_from_flat(&ws.tmp_joint_momentum, 0);
                let joint_m1 = vec6_from_flat(&ws.tmp_joint_momentum, 1);
                let joint_force = add6(joint_m1, hat_adj_wrench_vec6(v0, joint_m0));
                ws.joint_torque[j] = dot3(
                    self.axis[j],
                    [joint_force[0], joint_force[1], joint_force[2]],
                );
            } else {
                ws.joint_torque[j] = 0.0;
            }
        }
    }

    pub(crate) fn dynamics_cmtm_minimal_into(
        &self,
        motion: &[f64],
        dynamics_order: usize,
        gravity: [f64; 3],
        ws: &mut DynamicsCmtmWorkspace,
    ) {
        if dynamics_order != 1 || !gravity_is_zero(gravity) {
            self.dynamics_cmtm_into(motion, dynamics_order, gravity, ws);
            return;
        }

        ws.clear_minimal();
        self.kinematics_cmtm_into(motion, 3, &mut ws.cmtm);

        for link_id in 1..self.link_num {
            let r = mat3_from_flat(&ws.cmtm.fast_r, link_id);
            let rt = mat3_transpose(r);
            let v_world = [
                flat3(&ws.cmtm.fast_w, link_id),
                flat3(&ws.cmtm.fast_lin_v, link_id),
            ];
            let a_world = [
                flat3(&ws.cmtm.fast_alpha, link_id),
                flat3(&ws.cmtm.fast_lin_a, link_id),
            ];
            let v_local = [mat3_vec(rt, v_world[0]), mat3_vec(rt, v_world[1])];
            let a_local_ang = mat3_vec(rt, a_world[0]);
            let a_local_lin = sub3(mat3_vec(rt, a_world[1]), cross(v_local[0], v_local[1]));
            let a_local = [a_local_ang, a_local_lin];
            let momentum = mat6_vec(self.link_inertia[link_id], v_local);
            let inertial = mat6_vec(self.link_inertia[link_id], a_local);
            let force_local = [
                add3(
                    inertial[0],
                    add3(
                        cross(v_local[0], momentum[0]),
                        cross(v_local[1], momentum[1]),
                    ),
                ),
                add3(inertial[1], cross(v_local[0], momentum[1])),
            ];
            set_force(
                &mut ws.link_force,
                link_id,
                mat3_vec(r, force_local[0]),
                mat3_vec(r, force_local[1]),
            );
        }

        for j in (0..self.joint_num).rev() {
            let parent = self.parent_link[j];
            let child = self.child_link[j];
            if self.q_index[j] >= 0 {
                let parent_r = mat3_from_flat(&ws.cmtm.fast_r, parent);
                let axis_world = mat3_vec(mat3_mul(parent_r, self.origin_r[j]), self.axis[j]);
                ws.joint_torque[j] = dot3(axis_world, force_torque(&ws.link_force, child));
            }
            let rel = sub3(
                flat3(&ws.cmtm.fast_p, child),
                flat3(&ws.cmtm.fast_p, parent),
            );
            add_shifted_force_parent(&mut ws.link_force, parent, child, rel);
        }
    }
}
