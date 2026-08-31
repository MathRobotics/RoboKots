use crate::spatial::*;
use crate::cmtm_generic::{set_tangent_cmvecs, tangent_cmvecs, tangent_cmtm_vecs, tangent_mat4};
use crate::types::RustCompiledRobot;
use crate::workspace::{DynamicsCmtmTangentWorkspace, DynamicsCmtmWorkspace};

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

/// Analytic directional derivative of moving-frame gravity force series.
#[allow(clippy::too_many_arguments)]
fn gravity_force_series_tangent_into(
    link_mat: [[f64; 4]; 4],
    dlink_mat: [[f64; 4]; 4],
    link_vel: &[f64],
    dlink_vel: &[f64],
    inertia: [[f64; 6]; 6],
    gravity: [f64; 3],
    order: usize,
    local_gravity: &mut [f64],
    dlocal_gravity: &mut [f64],
    out: &mut [f64],
) {
    if order == 0 { return; }
    let rt = mat3_transpose(mat3_from_mat4(link_mat));
    let drt = mat3_transpose(mat3_from_mat4(dlink_mat));
    local_gravity[..3].copy_from_slice(&mat3_vec(rt, gravity));
    dlocal_gravity[..3].copy_from_slice(&mat3_vec(drt, gravity));
    for n in 1..order {
        let mut value = [0.0; 3];
        let mut dvalue = [0.0; 3];
        for k in 0..n {
            let omega = vec6_from_flat(link_vel, k);
            let domega = vec6_from_flat(dlink_vel, k);
            let g = [local_gravity[(n - 1 - k) * 3], local_gravity[(n - 1 - k) * 3 + 1], local_gravity[(n - 1 - k) * 3 + 2]];
            let dg = [dlocal_gravity[(n - 1 - k) * 3], dlocal_gravity[(n - 1 - k) * 3 + 1], dlocal_gravity[(n - 1 - k) * 3 + 2]];
            let c = binomial(n - 1, k);
            for i in 0..3 {
                value[i] -= c * cross([omega[0], omega[1], omega[2]], g)[i];
                dvalue[i] -= c * (cross([domega[0], domega[1], domega[2]], g)[i] + cross([omega[0], omega[1], omega[2]], dg)[i]);
            }
        }
        local_gravity[n * 3..(n + 1) * 3].copy_from_slice(&value);
        dlocal_gravity[n * 3..(n + 1) * 3].copy_from_slice(&dvalue);
    }
    for n in 0..order {
        let dg = [dlocal_gravity[n * 3], dlocal_gravity[n * 3 + 1], dlocal_gravity[n * 3 + 2]];
        let value = mat6_vec6(inertia, [0.0, 0.0, 0.0, dg[0], dg[1], dg[2]]);
        for i in 0..6 { out[n * 6 + i] = -value[i]; }
    }
}

/// Reverse-mode counterpart of [`gravity_force_series_into`].
///
/// The gravity series is a world-fixed vector represented in a moving link
/// frame.  `local_gravity` must be the primal series produced by the forward
/// routine; `local_gravity_bar` is scratch space and is cleared here.  The
/// resulting cotangents are accumulated into the raw link pose matrix and
/// raw spatial-velocity series (only its angular entries participate).
///
/// Keeping the recurrence explicit is important for high-order VJPs: a
/// basis-tangent implementation would evaluate this O(order * input_dim)
/// times, whereas this visits each binomial term once per output cotangent.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gravity_force_series_reverse_accumulate_into(
    link_vel: &[f64],
    inertia: [[f64; 6]; 6],
    gravity: [f64; 3],
    order: usize,
    local_gravity: &[f64],
    force_bar: &[f64],
    link_mat_bar: &mut [[f64; 4]; 4],
    link_vel_bar: &mut [f64],
    local_gravity_bar: &mut [f64],
) {
    if order == 0 {
        return;
    }
    debug_assert!(link_vel.len() >= order.saturating_sub(1) * 6);
    debug_assert!(local_gravity.len() >= order * 3);
    debug_assert!(force_bar.len() >= order * 6);
    debug_assert!(link_vel_bar.len() >= order.saturating_sub(1) * 6);
    debug_assert!(local_gravity_bar.len() >= order * 3);

    local_gravity_bar[..order * 3].fill(0.0);

    // f_n = -I [0, g_n].  Do not assume that the supplied inertia is exactly
    // symmetric: using the literal transpose makes this the adjoint of the
    // forward code for every 6x6 input matrix.
    for n in 0..order {
        for g_component in 0..3 {
            let mut value = 0.0;
            for force_component in 0..6 {
                value -= inertia[force_component][3 + g_component]
                    * force_bar[n * 6 + force_component];
            }
            local_gravity_bar[n * 3 + g_component] += value;
        }
    }

    // g_n = -sum_k C(n-1,k) (omega_k x g_(n-1-k)).
    // Process descending n so that every contribution to an earlier gravity
    // derivative is present before its own recurrence is reversed.
    for n in (1..order).rev() {
        let y = [
            local_gravity_bar[n * 3],
            local_gravity_bar[n * 3 + 1],
            local_gravity_bar[n * 3 + 2],
        ];
        for k in 0..n {
            let coefficient = -binomial(n - 1, k);
            let omega = vec6_from_flat(link_vel, k);
            let g_index = n - 1 - k;
            let g = [
                local_gravity[g_index * 3],
                local_gravity[g_index * 3 + 1],
                local_gravity[g_index * 3 + 2],
            ];

            // y . (omega x g) = omega . (g x y) = g . (y x omega).
            let omega_bar = scale3(cross(g, y), coefficient);
            let g_bar = scale3(cross(y, [omega[0], omega[1], omega[2]]), coefficient);
            for component in 0..3 {
                link_vel_bar[k * 6 + component] += omega_bar[component];
                local_gravity_bar[g_index * 3 + component] += g_bar[component];
            }
        }
    }

    // g_0 = R^T gravity.  Matrix entry R[r,c] has derivative gravity[r] in
    // local component c.  Translation and the homogeneous row do not enter.
    let g0_bar = [
        local_gravity_bar[0],
        local_gravity_bar[1],
        local_gravity_bar[2],
    ];
    for r in 0..3 {
        for c in 0..3 {
            link_mat_bar[r][c] += gravity[r] * g0_bar[c];
        }
    }

}

impl RustCompiledRobot {
    /// True reverse-mode VJP for the complete local CMTM dynamics output.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dynamics_cmtm_reverse_into(
        &self,
        motion: &[f64],
        link_momentum_cotangent: &[f64],
        link_force_cotangent: &[f64],
        joint_momentum_cotangent: &[f64],
        joint_force_cotangent: &[f64],
        torque_cotangent: &[f64],
        dynamics_order: usize,
        gravity: [f64; 3],
        rhs_cols: usize,
        kinetic_energy_cotangent: Option<&[f64]>,
        primal: &mut DynamicsCmtmWorkspace,
        out: &mut [f64],
    ) {
        let kin_order = dynamics_order + 2;
        let momentum_order = dynamics_order + 1;
        let input_len = self.dof * kin_order;
        self.dynamics_cmtm_into(motion, dynamics_order, gravity, primal);
        out.fill(0.0);
        let vec_len = (kin_order - 1) * 6;
        let momentum_len = momentum_order * 6;
        let force_len = dynamics_order * 6;
        let gravity_enabled = !gravity_is_zero(gravity);

        for rhs in 0..rhs_cols {
            let mut link_mat_bar = vec![[[0.0; 4]; 4]; self.link_num];
            let mut joint_mat_bar = vec![[[0.0; 4]; 4]; self.joint_num];
            let mut link_vec_bar = vec![0.0; self.link_num * vec_len];
            let mut joint_vec_bar = vec![0.0; self.joint_num * vec_len];
            let mut link_momentum_bar = vec![0.0; self.link_num * momentum_len];
            let mut joint_momentum_bar = vec![0.0; self.joint_num * momentum_len];
            let mut link_force_bar = vec![0.0; self.link_num * force_len];
            let mut joint_force_bar = vec![0.0; self.joint_num * force_len];
            let mut joint_gravity_bar = vec![0.0; self.joint_num * force_len];

            for link in 0..self.link_num {
                for i in 0..momentum_len {
                    link_momentum_bar[link * momentum_len + i] = link_momentum_cotangent[(link * momentum_len + i) * rhs_cols + rhs];
                }
                for i in 0..force_len {
                    link_force_bar[link * force_len + i] = link_force_cotangent[(link * force_len + i) * rhs_cols + rhs];
                }
            }
            for joint in 0..self.joint_num {
                for i in 0..momentum_len {
                    joint_momentum_bar[joint * momentum_len + i] = joint_momentum_cotangent[(joint * momentum_len + i) * rhs_cols + rhs];
                }
                for i in 0..force_len {
                    joint_force_bar[joint * force_len + i] = joint_force_cotangent[(joint * force_len + i) * rhs_cols + rhs];
                }
            }

            // torque = axis dot the angular part of joint force.
            for joint in 0..self.joint_num {
                if self.q_index[joint] < 0 {
                    // Fixed-joint torque rows are deliberately zero in the
                    // public CMTM API, matching the primal/tangent kernels.
                    continue;
                }
                for time in 0..dynamics_order {
                    let lambda = torque_cotangent[(joint * dynamics_order + time) * rhs_cols + rhs];
                    for c in 0..3 {
                        joint_force_bar[(joint * force_len + time * 6) + c] += self.axis[joint][c] * lambda;
                    }
                }
            }

            // The primal dynamics walks leaves-to-root; reverse it from the
            // root toward leaves so a parent transport contribution reaches
            // the child before that child is processed.
            for joint in 0..self.joint_num {
                let child = self.child_link[joint];
                let link_vec = cmtm_vecs_slice(&primal.cmtm.link_vecs, child, kin_order);
                let joint_momentum = cmvec_slice(&primal.joint_momentum, joint, momentum_order);
                let jf_start = joint * force_len;
                let jm_start = joint * momentum_len;
                let lv_start = child * vec_len;

                let joint_force_seed = joint_force_bar[jf_start..jf_start + force_len].to_vec();
                force_from_velocity_momentum_reverse_accumulate_into(
                    link_vec, joint_momentum, &joint_force_seed, dynamics_order, &primal.factorial,
                    &mut link_vec_bar[lv_start..lv_start + vec_len],
                    &mut joint_momentum_bar[jm_start..jm_start + momentum_len],
                );
                if gravity_enabled {
                    for i in 0..force_len {
                        joint_gravity_bar[jf_start + i] += joint_force_seed[i];
                    }
                    let joint_gravity_seed = joint_gravity_bar[jf_start..jf_start + force_len].to_vec();
                    let local_gravity = &primal.link_local_gravity[child * dynamics_order * 3..(child + 1) * dynamics_order * 3];
                    let mut local_gravity_bar = vec![0.0; dynamics_order * 3];
                    gravity_force_series_reverse_accumulate_into(
                        link_vec, self.link_inertia[child], gravity, dynamics_order, local_gravity,
                        &joint_gravity_seed, &mut link_mat_bar[child],
                        &mut link_vec_bar[lv_start..lv_start + vec_len], &mut local_gravity_bar,
                    );
                    for &child_joint in &self.link_child_joints[child] {
                        let origin = mat4_from_rot_pos(self.origin_r[child_joint], self.origin_p[child_joint]);
                        let rel_mat = mat4_mul(origin, mat4_from_flat(&primal.cmtm.joint_mat, child_joint));
                        let rel_vecs = cmtm_vecs_slice(&primal.cmtm.joint_vecs, child_joint, kin_order);
                        let child_gravity = cmvec_slice(&primal.joint_gravity_force, child_joint, dynamics_order);
                        let mut rel_mat_bar = [[0.0; 4]; 4];
                        let mut a = vec![[[0.0; 3]; 3]; dynamics_order];
                        let mut c = vec![[[0.0; 3]; 3]; dynamics_order];
                        let mut ab = vec![[[0.0; 3]; 3]; dynamics_order];
                        let mut cb = vec![[[0.0; 3]; 3]; dynamics_order];
                        let mut scaled = vec![0.0; dynamics_order.saturating_sub(1) * 6];
                        cmtm_accumulate_mat_adj_wrench_series_reverse_accumulate_into(
                            rel_mat, &rel_vecs[..dynamics_order.saturating_sub(1) * 6], child_gravity,
                            &joint_gravity_seed, dynamics_order, &primal.factorial, &mut scaled,
                            &mut a, &mut c, &mut ab, &mut cb,
                            &mut joint_gravity_bar[child_joint * force_len..(child_joint + 1) * force_len],
                            &mut joint_vec_bar[child_joint * vec_len..(child_joint + 1) * vec_len], &mut rel_mat_bar,
                        );
                        for r in 0..4 { for col in 0..4 {
                            joint_mat_bar[child_joint][r][col] += (0..4).map(|k| origin[k][r] * rel_mat_bar[k][col]).sum::<f64>();
                        }}
                    }
                }

                let joint_momentum_seed = joint_momentum_bar[jm_start..jm_start + momentum_len].to_vec();
                for i in 0..momentum_len { link_momentum_bar[child * momentum_len + i] += joint_momentum_seed[i]; }
                for &child_joint in &self.link_child_joints[child] {
                    let origin = mat4_from_rot_pos(self.origin_r[child_joint], self.origin_p[child_joint]);
                    let rel_mat = mat4_mul(origin, mat4_from_flat(&primal.cmtm.joint_mat, child_joint));
                    let rel_vecs = cmtm_vecs_slice(&primal.cmtm.joint_vecs, child_joint, kin_order);
                    let child_momentum = cmvec_slice(&primal.joint_momentum, child_joint, momentum_order);
                    let mut rel_mat_bar = [[0.0; 4]; 4];
                    let mut a = vec![[[0.0; 3]; 3]; momentum_order];
                    let mut c = vec![[[0.0; 3]; 3]; momentum_order];
                    let mut ab = vec![[[0.0; 3]; 3]; momentum_order];
                    let mut cb = vec![[[0.0; 3]; 3]; momentum_order];
                    let mut scaled = vec![0.0; (momentum_order - 1) * 6];
                    cmtm_accumulate_mat_adj_wrench_series_reverse_accumulate_into(
                        rel_mat, &rel_vecs[..(momentum_order - 1) * 6], child_momentum,
                        &joint_momentum_seed, momentum_order, &primal.factorial, &mut scaled,
                        &mut a, &mut c, &mut ab, &mut cb,
                        &mut joint_momentum_bar[child_joint * momentum_len..(child_joint + 1) * momentum_len],
                        &mut joint_vec_bar[child_joint * vec_len..(child_joint + 1) * vec_len], &mut rel_mat_bar,
                    );
                    for r in 0..4 { for col in 0..4 {
                        joint_mat_bar[child_joint][r][col] += (0..4).map(|k| origin[k][r] * rel_mat_bar[k][col]).sum::<f64>();
                    }}
                }

                let lf_start = child * force_len;
                let link_force_seed = link_force_bar[lf_start..lf_start + force_len].to_vec();
                let link_momentum = cmvec_slice(&primal.link_momentum, child, momentum_order);
                force_from_velocity_momentum_reverse_accumulate_into(
                    link_vec, link_momentum, &link_force_seed, dynamics_order, &primal.factorial,
                    &mut link_vec_bar[lv_start..lv_start + vec_len],
                    &mut link_momentum_bar[child * momentum_len..(child + 1) * momentum_len],
                );
                if gravity_enabled {
                    let local_gravity = &primal.link_local_gravity[child * dynamics_order * 3..(child + 1) * dynamics_order * 3];
                    let mut local_gravity_bar = vec![0.0; dynamics_order * 3];
                    gravity_force_series_reverse_accumulate_into(
                        link_vec, self.link_inertia[child], gravity, dynamics_order, local_gravity,
                        &link_force_seed, &mut link_mat_bar[child],
                        &mut link_vec_bar[lv_start..lv_start + vec_len], &mut local_gravity_bar,
                    );
                }
                for time in 0..momentum_order {
                    let bar = mat6_transpose_vec6(self.link_inertia[child], vec6_from_flat(&link_momentum_bar[child * momentum_len..], time));
                    for c in 0..6 { link_vec_bar[lv_start + time * 6 + c] += bar[c]; }
                }
            }

            let mut link_mat_flat = vec![0.0; self.link_num * 16];
            let mut joint_mat_flat = vec![0.0; self.joint_num * 16];
            for link in 0..self.link_num { for r in 0..4 { for c in 0..4 { link_mat_flat[link * 16 + r * 4 + c] = link_mat_bar[link][r][c]; }}}
            for joint in 0..self.joint_num { for r in 0..4 { for c in 0..4 { joint_mat_flat[joint * 16 + r * 4 + c] = joint_mat_bar[joint][r][c]; }}}
            if let Some(energy_cotangent) = kinetic_energy_cotangent {
                let lambda = energy_cotangent[rhs];
                if lambda != 0.0 {
                    for link in 0..self.link_num {
                        let velocity = vec6_from_flat(
                            cmtm_vecs_slice(&primal.cmtm.link_vecs, link, kin_order), 0,
                        );
                        let iv = mat6_vec6(self.link_inertia[link], velocity);
                        let itv = mat6_transpose_vec6(self.link_inertia[link], velocity);
                        let lv_start = link * vec_len;
                        for c in 0..6 {
                            link_vec_bar[lv_start + c] += 0.5 * (iv[c] + itv[c]) * lambda;
                        }
                    }
                }
            }
            let mut motion_bar = vec![0.0; input_len];
            self.kinematics_cmtm_outward_reverse_into(
                motion, kin_order, &link_mat_flat, &link_vec_bar, &joint_mat_flat, &joint_vec_bar,
                1, &mut primal.cmtm, &mut motion_bar,
            );
            for input in 0..input_len { out[input * rhs_cols + rhs] = motion_bar[input]; }
        }
    }

    /// Torque-only adapter for the public torque-series VJP API.
    pub(crate) fn dynamics_joint_torque_series_reverse_into(
        &self,
        motion: &[f64], torque_cotangent: &[f64], dynamics_order: usize,
        gravity: [f64; 3], rhs_cols: usize, primal: &mut DynamicsCmtmWorkspace,
        out: &mut [f64],
    ) {
        let momentum_len = self.link_num * (dynamics_order + 1) * 6 * rhs_cols;
        let link_force_len = self.link_num * dynamics_order * 6 * rhs_cols;
        let joint_momentum_len = self.joint_num * (dynamics_order + 1) * 6 * rhs_cols;
        let joint_force_len = self.joint_num * dynamics_order * 6 * rhs_cols;
        self.dynamics_cmtm_reverse_into(
            motion, &vec![0.0; momentum_len], &vec![0.0; link_force_len],
            &vec![0.0; joint_momentum_len], &vec![0.0; joint_force_len],
            torque_cotangent, dynamics_order, gravity, rhs_cols, None, primal, out,
        );
    }

    /// Fuse torque-series and kinetic-energy cotangents into one dynamics
    /// reverse pass.  Kinetic energy seeds the velocity slot of the same
    /// final kinematics reverse used by the dynamics recurrence.
    pub(crate) fn dynamics_joint_torque_series_energy_reverse_into(
        &self,
        motion: &[f64], torque_cotangent: &[f64], energy_cotangent: &[f64],
        dynamics_order: usize, gravity: [f64; 3], rhs_cols: usize,
        primal: &mut DynamicsCmtmWorkspace, out: &mut [f64],
    ) {
        let momentum_len = self.link_num * (dynamics_order + 1) * 6 * rhs_cols;
        let link_force_len = self.link_num * dynamics_order * 6 * rhs_cols;
        let joint_momentum_len = self.joint_num * (dynamics_order + 1) * 6 * rhs_cols;
        let joint_force_len = self.joint_num * dynamics_order * 6 * rhs_cols;
        self.dynamics_cmtm_reverse_into(
            motion, &vec![0.0; momentum_len], &vec![0.0; link_force_len],
            &vec![0.0; joint_momentum_len], &vec![0.0; joint_force_len],
            torque_cotangent, dynamics_order, gravity, rhs_cols,
            Some(energy_cotangent), primal, out,
        );
    }

    /// Differentiate local link momentum and force series after the CMTM
    /// kinematics tangent has been propagated.  Joint wrench accumulation and
    /// gravity are intentionally layered above this primitive.
    #[allow(dead_code)]
    pub(crate) fn dynamics_cmtm_link_tangent_into(
        &self,
        motion: &[f64],
        motion_tangent: &[f64],
        dynamics_order: usize,
        gravity: [f64; 3],
        primal: &mut DynamicsCmtmWorkspace,
        tangent: &mut DynamicsCmtmTangentWorkspace,
    ) {
        let kin_order = dynamics_order + 2;
        let momentum_order = dynamics_order + 1;
        self.dynamics_cmtm_into(motion, dynamics_order, gravity, primal);
        self.kinematics_cmtm_tangent_into(
            motion,
            motion_tangent,
            kin_order,
            &mut primal.cmtm,
            tangent,
        );
        for link in 0..self.link_num {
            let vel = cmtm_vecs_slice(&primal.cmtm.link_vecs, link, kin_order);
            for rhs_col in 0..tangent.rhs_cols {
                let dvel = tangent_cmtm_vecs(
                    &tangent.link_vecs,
                    link,
                    kin_order,
                    tangent.rhs_cols,
                    rhs_col,
                );
                let mut dmomentum = vec![0.0; momentum_order * 6];
                for k in 0..momentum_order {
                    set_vec6_flat(
                        &mut dmomentum,
                        k,
                        mat6_vec6(self.link_inertia[link], vec6_from_flat(&dvel, k)),
                    );
                }
                set_tangent_cmvecs(
                    &mut tangent.link_momentum,
                    link,
                    momentum_order,
                    tangent.rhs_cols,
                    rhs_col,
                    &dmomentum,
                );
                if dynamics_order > 0 {
                    let momentum = cmvec_slice(&primal.link_momentum, link, momentum_order);
                    let mut dforce = vec![0.0; dynamics_order * 6];
                    force_from_velocity_momentum_tangent_into(
                        vel,
                        &dvel,
                        momentum,
                        &dmomentum,
                        dynamics_order,
                        &primal.factorial,
                        &mut dforce,
                    );
                    if !gravity_is_zero(gravity) {
                        let dmat = tangent_mat4(&tangent.link_mat, link, tangent.rhs_cols, rhs_col);
                        let mut dgravity = vec![0.0; dynamics_order * 6];
                        let mut local_g = vec![0.0; dynamics_order * 3];
                        let mut dlocal_g = vec![0.0; dynamics_order * 3];
                        gravity_force_series_tangent_into(
                            mat4_from_flat(&primal.cmtm.link_mat, link), dmat, vel, &dvel,
                            self.link_inertia[link], gravity, dynamics_order,
                            &mut local_g, &mut dlocal_g, &mut dgravity,
                        );
                        for i in 0..dynamics_order * 6 { dforce[i] += dgravity[i]; }
                    }
                    set_tangent_cmvecs(
                        &mut tangent.link_force,
                        link,
                        dynamics_order,
                        tangent.rhs_cols,
                        rhs_col,
                        &dforce,
                    );
                }
            }
        }
        // Reverse-tree accumulation mirrors dynamics_cmtm_into.  At this
        // layer gravity transport is added separately; this block handles the
        // inertial joint momentum and torque path.
        for j in (0..self.joint_num).rev() {
            let child = self.child_link[j];
            let child_joint_ids = &self.link_child_joints[child];
            let vel = cmtm_vecs_slice(&primal.cmtm.link_vecs, child, kin_order);
            for rhs_col in 0..tangent.rhs_cols {
                let mut d_joint_momentum = tangent_cmvecs(
                    &tangent.link_momentum, child, momentum_order, tangent.rhs_cols, rhs_col,
                );
                for &child_joint in child_joint_ids {
                    let joint_mat = mat4_from_flat(&primal.cmtm.joint_mat, child_joint);
                    let origin = mat4_from_rot_pos(self.origin_r[child_joint], self.origin_p[child_joint]);
                    let rel_mat = mat4_mul(origin, joint_mat);
                    let drel_mat = mat4_mul(
                        origin,
                        tangent_mat4(&tangent.joint_mat, child_joint, tangent.rhs_cols, rhs_col),
                    );
                    let rel_vecs = cmtm_vecs_slice(&primal.cmtm.joint_vecs, child_joint, kin_order);
                    let drel_vecs = tangent_cmtm_vecs(
                        &tangent.joint_vecs, child_joint, kin_order, tangent.rhs_cols, rhs_col,
                    );
                    let rhs = cmvec_slice(&primal.joint_momentum, child_joint, momentum_order);
                    let drhs = tangent_cmvecs(
                        &tangent.joint_momentum, child_joint, momentum_order, tangent.rhs_cols, rhs_col,
                    );
                    let mut blocks = vec![[[0.0; 6]; 6]; momentum_order];
                    let mut dblocks = vec![[[0.0; 6]; 6]; momentum_order];
                    let mut transported = vec![0.0; momentum_order * 6];
                    let mut dtransported = vec![0.0; momentum_order * 6];
                    cmtm_apply_mat_adj_wrench_tangent_into(
                        rel_mat, &rel_vecs[..(momentum_order - 1) * 6], drel_mat, &drel_vecs[..(momentum_order - 1) * 6],
                        rhs, &drhs, momentum_order, &primal.factorial,
                        &mut blocks, &mut dblocks, &mut transported, &mut dtransported,
                    );
                    for i in 0..momentum_order * 6 { d_joint_momentum[i] += dtransported[i]; }
                }
                set_tangent_cmvecs(&mut tangent.joint_momentum, j, momentum_order, tangent.rhs_cols, rhs_col, &d_joint_momentum);
                let mut dgravity_force = vec![0.0; dynamics_order * 6];
                if !gravity_is_zero(gravity) {
                    let dvel = tangent_cmtm_vecs(&tangent.link_vecs, child, kin_order, tangent.rhs_cols, rhs_col);
                    let mut local_g = vec![0.0; dynamics_order * 3];
                    let mut dlocal_g = vec![0.0; dynamics_order * 3];
                    gravity_force_series_tangent_into(
                        mat4_from_flat(&primal.cmtm.link_mat, child),
                        tangent_mat4(&tangent.link_mat, child, tangent.rhs_cols, rhs_col),
                        vel, &dvel, self.link_inertia[child], gravity, dynamics_order,
                        &mut local_g, &mut dlocal_g, &mut dgravity_force,
                    );
                    for &child_joint in child_joint_ids {
                        let origin = mat4_from_rot_pos(self.origin_r[child_joint], self.origin_p[child_joint]);
                        let rel_mat = mat4_mul(origin, mat4_from_flat(&primal.cmtm.joint_mat, child_joint));
                        let drel_mat = mat4_mul(origin, tangent_mat4(&tangent.joint_mat, child_joint, tangent.rhs_cols, rhs_col));
                        let rel_vecs = cmtm_vecs_slice(&primal.cmtm.joint_vecs, child_joint, kin_order);
                        let drel_vecs = tangent_cmtm_vecs(&tangent.joint_vecs, child_joint, kin_order, tangent.rhs_cols, rhs_col);
                        let rhs = cmvec_slice(&primal.joint_gravity_force, child_joint, dynamics_order);
                        let drhs = tangent_cmvecs(&tangent.joint_gravity_force, child_joint, dynamics_order, tangent.rhs_cols, rhs_col);
                        let mut blocks = vec![[[0.0; 6]; 6]; dynamics_order];
                        let mut dblocks = vec![[[0.0; 6]; 6]; dynamics_order];
                        let mut transported = vec![0.0; dynamics_order * 6];
                        let mut dtransported = vec![0.0; dynamics_order * 6];
                        cmtm_apply_mat_adj_wrench_tangent_into(rel_mat, &rel_vecs[..dynamics_order.saturating_sub(1) * 6], drel_mat, &drel_vecs[..dynamics_order.saturating_sub(1) * 6], rhs, &drhs, dynamics_order, &primal.factorial, &mut blocks, &mut dblocks, &mut transported, &mut dtransported);
                        for i in 0..dynamics_order * 6 { dgravity_force[i] += dtransported[i]; }
                    }
                    set_tangent_cmvecs(&mut tangent.joint_gravity_force, j, dynamics_order, tangent.rhs_cols, rhs_col, &dgravity_force);
                }
                let mut dforce = vec![0.0; dynamics_order * 6];
                force_from_velocity_momentum_tangent_into(
                    vel, &tangent_cmtm_vecs(&tangent.link_vecs, child, kin_order, tangent.rhs_cols, rhs_col),
                    cmvec_slice(&primal.joint_momentum, j, momentum_order), &d_joint_momentum,
                    dynamics_order, &primal.factorial, &mut dforce,
                );
                for i in 0..dynamics_order * 6 { dforce[i] += dgravity_force[i]; }
                set_tangent_cmvecs(&mut tangent.joint_force, j, dynamics_order, tangent.rhs_cols, rhs_col, &dforce);
                if self.q_index[j] >= 0 {
                    for k in 0..dynamics_order {
                        tangent.joint_torque[(j * dynamics_order + k) * tangent.rhs_cols + rhs_col] =
                            dot3(self.axis[j], [dforce[k * 6], dforce[k * 6 + 1], dforce[k * 6 + 2]]);
                    }
                }
            }
        }
    }
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
                    let gravity_offset = child * dynamics_order * 3;
                    ws.link_local_gravity[gravity_offset..gravity_offset + dynamics_order * 3]
                        .copy_from_slice(&ws.tmp_local_gravity[..dynamics_order * 3]);
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
                let gravity_offset = world * dynamics_order * 3;
                ws.link_local_gravity[gravity_offset..gravity_offset + dynamics_order * 3]
                    .copy_from_slice(&ws.tmp_local_gravity[..dynamics_order * 3]);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_force_series_reverse_satisfies_directional_duality() {
        const ORDER: usize = 5;
        let link_mat = [
            [0.81, -0.33, 0.48, 0.2],
            [0.41, 0.90, -0.12, -0.1],
            [-0.42, 0.28, 0.86, 0.4],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let dlink_mat = [
            [0.12, -0.07, 0.03, 0.0],
            [-0.05, 0.09, 0.11, 0.0],
            [0.02, -0.04, 0.08, 0.0],
            [0.0; 4],
        ];
        let link_vel = [
            0.3, -0.4, 0.2, 0.1, 0.2, -0.1,
            -0.2, 0.1, 0.5, 0.0, -0.1, 0.2,
            0.4, 0.2, -0.3, 0.1, 0.0, -0.2,
            -0.1, 0.3, 0.2, 0.0, 0.0, 0.1,
        ];
        let dlink_vel = [
            -0.2, 0.1, 0.3, 0.0, 0.0, 0.0,
            0.4, -0.1, 0.2, 0.0, 0.0, 0.0,
            0.1, 0.2, -0.4, 0.0, 0.0, 0.0,
            -0.3, 0.2, 0.1, 0.0, 0.0, 0.0,
        ];
        let inertia = [
            [2.0, 0.1, -0.2, 0.3, 0.0, -0.1],
            [0.1, 1.7, 0.2, -0.1, 0.4, 0.0],
            [-0.2, 0.2, 1.9, 0.0, -0.2, 0.5],
            [0.3, -0.1, 0.0, 3.0, 0.2, -0.1],
            [0.0, 0.4, -0.2, 0.2, 2.7, 0.3],
            [-0.1, 0.0, 0.5, -0.1, 0.3, 2.5],
        ];
        let gravity = [0.4, -9.81, 1.2];
        let force_bar = [
            0.2, -0.1, 0.4, -0.3, 0.5, 0.1,
            -0.2, 0.3, 0.1, 0.4, -0.5, 0.2,
            0.1, 0.2, -0.3, 0.5, 0.4, -0.2,
            -0.4, 0.1, 0.2, -0.1, 0.3, 0.5,
            0.3, -0.5, 0.2, 0.1, -0.2, 0.4,
        ];
        let mut local_g = vec![0.0; ORDER * 3];
        let mut force = vec![0.0; ORDER * 6];
        gravity_force_series_into(link_mat, &link_vel, inertia, gravity, ORDER, &mut local_g, &mut force);
        let mut dlocal_g = vec![0.0; ORDER * 3];
        let mut dforce = vec![0.0; ORDER * 6];
        gravity_force_series_tangent_into(
            link_mat, dlink_mat, &link_vel, &dlink_vel, inertia, gravity, ORDER,
            &mut local_g.clone(), &mut dlocal_g, &mut dforce,
        );
        let mut link_mat_bar = [[0.0; 4]; 4];
        let mut link_vel_bar = vec![0.0; (ORDER - 1) * 6];
        let mut local_g_bar = vec![0.0; ORDER * 3];
        gravity_force_series_reverse_accumulate_into(
            &link_vel, inertia, gravity, ORDER, &local_g, &force_bar,
            &mut link_mat_bar, &mut link_vel_bar, &mut local_g_bar,
        );
        let lhs: f64 = force_bar.iter().zip(&dforce).map(|(a, b)| a * b).sum();
        let mut rhs = 0.0;
        for r in 0..4 { for c in 0..4 { rhs += link_mat_bar[r][c] * dlink_mat[r][c]; } }
        rhs += link_vel_bar.iter().zip(&dlink_vel).map(|(a, b)| a * b).sum::<f64>();
        assert!((lhs - rhs).abs() < 1e-10, "lhs={lhs}, rhs={rhs}");
    }
}
