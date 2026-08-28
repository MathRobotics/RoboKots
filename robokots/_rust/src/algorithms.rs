use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::spatial::*;
use crate::types::RustCompiledRobot;
use crate::workspace::{AbaWorkspace, BulkDerivativeWorkspace, Workspace};

fn aba_dot(a: [f64; 6], b: [f64; 6]) -> f64 { (0..6).map(|i| a[i] * b[i]).sum() }
fn aba_add(a: [f64; 6], b: [f64; 6]) -> [f64; 6] { let mut o = a; for i in 0..6 { o[i] += b[i]; } o }
fn aba_scale(a: [f64; 6], s: f64) -> [f64; 6] { let mut o = a; for x in &mut o { *x *= s; } o }

fn aba_shift_motion_to_child(v: [f64; 6], rel: [f64; 3]) -> [f64; 6] {
    let linear = add3([v[3], v[4], v[5]], cross([v[0], v[1], v[2]], rel));
    [v[0], v[1], v[2], linear[0], linear[1], linear[2]]
}

fn aba_shift_force_to_parent(f: [f64; 6], rel: [f64; 3]) -> [f64; 6] {
    let n = [f[0], f[1], f[2]]; let force = [f[3], f[4], f[5]]; let nr = add3(n, cross(rel, force));
    [nr[0], nr[1], nr[2], force[0], force[1], force[2]]
}

fn aba_world_inertia(r: [[f64; 3]; 3], local: [[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut b = [[0.0; 6]; 6];
    for i in 0..3 { for j in 0..3 { b[i][j] = r[i][j]; b[i + 3][j + 3] = r[i][j]; }}
    mat6_mul(mat6_mul(b, local), mat6_transpose(b))
}

fn mat6_transpose(a: [[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut out = [[0.0; 6]; 6]; for i in 0..6 { for j in 0..6 { out[i][j] = a[j][i]; }} out
}

impl RustCompiledRobot {
    pub(crate) fn check_motion(&self, q: &[f64], v: &[f64], a: &[f64]) -> PyResult<()> {
        if q.len() != self.dof || v.len() != self.dof || a.len() != self.dof {
            return Err(PyValueError::new_err("q/v/a length must match robot dof"));
        }
        Ok(())
    }

    pub(crate) fn check_motion_batch(
        &self,
        q_shape: &[usize],
        v_shape: &[usize],
        a_shape: &[usize],
    ) -> PyResult<usize> {
        if q_shape.len() != 2 || v_shape.len() != 2 || a_shape.len() != 2 {
            return Err(PyValueError::new_err(
                "q/v/a batch shapes must be (batch, robot dof)",
            ));
        }
        if q_shape != v_shape || q_shape != a_shape {
            return Err(PyValueError::new_err("q/v/a batch shapes must match"));
        }
        if q_shape[1] != self.dof {
            return Err(PyValueError::new_err(
                "q/v/a batch last dimension must match robot dof",
            ));
        }
        Ok(q_shape[0])
    }

    pub(crate) fn joint_jacobians_vec(&self, q: &[f64]) -> Vec<f64> {
        let mut ws = Workspace::new(self);
        self.joint_jacobians_into(q, &mut ws);
        ws.jac
    }

    pub(crate) fn joint_jacobians_into(&self, q: &[f64], ws: &mut Workspace) {
        let mut zero = std::mem::take(&mut ws.zero_motion);
        if zero.len() != self.dof {
            zero.resize(self.dof, 0.0);
        }
        zero.fill(0.0);
        self.forward_kinematics_into(q, &zero, &zero, ws);
        ws.zero_motion = zero;
        ws.jac.fill(0.0);
        ws.active_axes.fill([0.0; 3]);
        ws.active_points.fill([0.0; 3]);

        for j in 0..self.joint_num {
            let qi = self.q_index[j];
            if qi < 0 {
                continue;
            }
            let qi = qi as usize;
            let parent = self.parent_link[j];
            let joint_r = mat3_mul(mat3_from_flat(&ws.r, parent), self.origin_r[j]);
            ws.active_axes[qi] = mat3_vec(joint_r, self.axis[j]);
            ws.active_points[qi] = add3(
                flat3(&ws.p, parent),
                mat3_vec(mat3_from_flat(&ws.r, parent), self.origin_p[j]),
            );
        }

        for link_id in 0..self.link_num {
            let link_p = flat3(&ws.p, link_id);
            for &qi in &self.link_ancestors[link_id] {
                let axis = ws.active_axes[qi];
                set_jac(&mut ws.jac, self.dof, link_id, 0, qi, axis[0]);
                set_jac(&mut ws.jac, self.dof, link_id, 1, qi, axis[1]);
                set_jac(&mut ws.jac, self.dof, link_id, 2, qi, axis[2]);
                let lin = cross(axis, sub3(link_p, ws.active_points[qi]));
                set_jac(&mut ws.jac, self.dof, link_id, 3, qi, lin[0]);
                set_jac(&mut ws.jac, self.dof, link_id, 4, qi, lin[1]);
                set_jac(&mut ws.jac, self.dof, link_id, 5, qi, lin[2]);
            }
        }
    }

    pub(crate) fn forward_kinematics_into(
        &self,
        q: &[f64],
        v: &[f64],
        a: &[f64],
        ws: &mut Workspace,
    ) {
        ws.clear_kinematics();
        set_eye3(&mut ws.r, 0);

        for j in 0..self.joint_num {
            let parent = self.parent_link[j];
            let child = self.child_link[j];
            let parent_r = mat3_from_flat(&ws.r, parent);
            let parent_p = flat3(&ws.p, parent);
            let parent_w = flat3(&ws.w, parent);
            let parent_lin_v = flat3(&ws.lin_v, parent);
            let parent_alpha = flat3(&ws.alpha, parent);
            let parent_lin_a = flat3(&ws.lin_a, parent);
            let joint_r0 = mat3_mul(parent_r, self.origin_r[j]);
            let joint_p = add3(parent_p, mat3_vec(parent_r, self.origin_p[j]));
            let qi = self.q_index[j];
            if qi >= 0 {
                let qi = qi as usize;
                let axis_world = mat3_vec(joint_r0, self.axis[j]);
                if self.is_prismatic[j] {
                    let child_p = add3(joint_p, scale3(axis_world, q[qi]));
                    let rel = sub3(child_p, parent_p);
                    set_mat3(&mut ws.r, child, joint_r0);
                    set_flat3(&mut ws.p, child, child_p);
                    set_flat3(&mut ws.w, child, parent_w);
                    set_flat3(
                        &mut ws.lin_v,
                        child,
                        add3(
                            add3(parent_lin_v, cross(parent_w, rel)),
                            scale3(axis_world, v[qi]),
                        ),
                    );
                    set_flat3(&mut ws.alpha, child, parent_alpha);
                    set_flat3(
                        &mut ws.lin_a,
                        child,
                        add3(
                            add3(
                                add3(parent_lin_a, cross(parent_alpha, rel)),
                                cross(parent_w, cross(parent_w, rel)),
                            ),
                            add3(
                                scale3(cross(parent_w, axis_world), 2.0 * v[qi]),
                                scale3(axis_world, a[qi]),
                            ),
                        ),
                    );
                    continue;
                }
                let rj = rot_axis(self.axis[j], q[qi]);
                set_mat3(&mut ws.r, child, mat3_mul(joint_r0, rj));
                set_flat3(&mut ws.p, child, joint_p);
                let rel = sub3(joint_p, parent_p);
                set_flat3(&mut ws.w, child, add3(parent_w, scale3(axis_world, v[qi])));
                set_flat3(
                    &mut ws.lin_v,
                    child,
                    add3(parent_lin_v, cross(parent_w, rel)),
                );
                set_flat3(
                    &mut ws.alpha,
                    child,
                    add3(
                        add3(parent_alpha, scale3(axis_world, a[qi])),
                        cross(parent_w, scale3(axis_world, v[qi])),
                    ),
                );
                set_flat3(
                    &mut ws.lin_a,
                    child,
                    add3(
                        add3(parent_lin_a, cross(parent_alpha, rel)),
                        cross(parent_w, cross(parent_w, rel)),
                    ),
                );
            } else {
                set_mat3(&mut ws.r, child, joint_r0);
                set_flat3(&mut ws.p, child, joint_p);
                let rel = sub3(joint_p, parent_p);
                set_flat3(&mut ws.w, child, parent_w);
                set_flat3(
                    &mut ws.lin_v,
                    child,
                    add3(parent_lin_v, cross(parent_w, rel)),
                );
                set_flat3(&mut ws.alpha, child, parent_alpha);
                set_flat3(
                    &mut ws.lin_a,
                    child,
                    add3(
                        add3(parent_lin_a, cross(parent_alpha, rel)),
                        cross(parent_w, cross(parent_w, rel)),
                    ),
                );
            }
        }
    }

    pub(crate) fn rnea_into(&self, q: &[f64], v: &[f64], a: &[f64], ws: &mut Workspace) {
        self.rnea_with_gravity_into(q, v, a, [0.0; 3], ws);
    }

    pub(crate) fn rnea_with_gravity_into(
        &self,
        q: &[f64],
        v: &[f64],
        a: &[f64],
        gravity: [f64; 3],
        ws: &mut Workspace,
    ) {
        self.forward_kinematics_into(q, v, a, ws);
        ws.forces.fill(0.0);
        ws.tau.fill(0.0);

        for link_id in 1..self.link_num {
            let r = mat3_from_flat(&ws.r, link_id);
            let rt = mat3_transpose(r);
            let v_world = [flat3(&ws.w, link_id), flat3(&ws.lin_v, link_id)];
            let a_world = [flat3(&ws.alpha, link_id), flat3(&ws.lin_a, link_id)];
            let v_local = [mat3_vec(rt, v_world[0]), mat3_vec(rt, v_world[1])];
            let a_local_ang = mat3_vec(rt, a_world[0]);
            // RNEA represents gravity as the opposite acceleration of the
            // inertial frame. `gravity` is expressed in the world frame.
            let a_local_lin = sub3(
                sub3(mat3_vec(rt, a_world[1]), mat3_vec(rt, gravity)),
                cross(v_local[0], v_local[1]),
            );
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
                &mut ws.forces,
                link_id,
                mat3_vec(r, force_local[0]),
                mat3_vec(r, force_local[1]),
            );
        }

        for j in (0..self.joint_num).rev() {
            let parent = self.parent_link[j];
            let child = self.child_link[j];
            let qi = self.q_index[j];
            if qi >= 0 {
                let parent_r = mat3_from_flat(&ws.r, parent);
                let axis_world = mat3_vec(mat3_mul(parent_r, self.origin_r[j]), self.axis[j]);
                let projected_force = if self.is_prismatic[j] {
                    force_force(&ws.forces, child)
                } else {
                    force_torque(&ws.forces, child)
                };
                ws.tau[qi as usize] = dot3(axis_world, projected_force);
            }
            let rel = sub3(flat3(&ws.p, child), flat3(&ws.p, parent));
            add_shifted_force_parent(&mut ws.forces, parent, child, rel);
        }
    }

    /// Fixed-base articulated-body algorithm in world spatial coordinates.
    /// This order-zero kernel is intentionally independent of CMTM series
    /// storage; `CmtmAbaWorkspace` can reuse these spatial recurrences later.
    pub(crate) fn aba_with_gravity_into(
        &self, q: &[f64], v: &[f64], tau: &[f64], gravity: [f64; 3], ws: &mut AbaWorkspace,
    ) -> Result<(), &'static str> {
        if q.len() != self.dof || v.len() != self.dof || tau.len() != self.dof { return Err("q/v/tau length must match robot dof"); }
        ws.zero.fill(0.0);
        self.forward_kinematics_into(q, v, &ws.zero, &mut ws.kinematics);
        ws.qdd.fill(0.0);
        for link in 0..self.link_num {
            let r = mat3_from_flat(&ws.kinematics.r, link);
            ws.ia[link] = aba_world_inertia(r, self.link_inertia[link]);
            let velocity = [
                ws.kinematics.w[link * 3], ws.kinematics.w[link * 3 + 1], ws.kinematics.w[link * 3 + 2],
                ws.kinematics.lin_v[link * 3], ws.kinematics.lin_v[link * 3 + 1], ws.kinematics.lin_v[link * 3 + 2],
            ];
            let momentum = mat6_vec6(ws.ia[link], velocity);
            let gravity_spatial = [0.0, 0.0, 0.0, gravity[0], gravity[1], gravity[2]];
            let gravity_force = mat6_vec6(ws.ia[link], gravity_spatial);
            let mut bias = hat_adj_wrench_vec6(velocity, momentum);
            for i in 0..6 { bias[i] -= gravity_force[i]; }
            ws.pa[link] = bias;
            let w = [velocity[0], velocity[1], velocity[2]];
            let lin_v = [velocity[3], velocity[4], velocity[5]];
            let alpha = [ws.kinematics.alpha[link * 3], ws.kinematics.alpha[link * 3 + 1], ws.kinematics.alpha[link * 3 + 2]];
            let lin_a = [ws.kinematics.lin_a[link * 3], ws.kinematics.lin_a[link * 3 + 1], ws.kinematics.lin_a[link * 3 + 2]];
            let spatial_lin = sub3(lin_a, cross(w, lin_v));
            ws.c[link] = [alpha[0], alpha[1], alpha[2], spatial_lin[0], spatial_lin[1], spatial_lin[2]];
        }
        for joint in (0..self.joint_num).rev() {
            let parent_r = mat3_from_flat(&ws.kinematics.r, self.parent_link[joint]);
            let axis = mat3_vec(mat3_mul(parent_r, self.origin_r[joint]), self.axis[joint]);
            ws.s[joint] = if self.is_prismatic[joint] { [0.0, 0.0, 0.0, axis[0], axis[1], axis[2]] } else { [axis[0], axis[1], axis[2], 0.0, 0.0, 0.0] };
        }
        // `forward_kinematics_into(a=0)` stores total spatial acceleration.
        // ABA needs the per-joint bias c_i in a_i = Xup a_parent + c_i + S qdd.
        for joint in (0..self.joint_num).rev() {
            let parent = self.parent_link[joint]; let child = self.child_link[joint];
            let rel = sub3(flat3(&ws.kinematics.p, child), flat3(&ws.kinematics.p, parent));
            let parent_total = ws.c[parent];
            let propagated = aba_shift_motion_to_child(parent_total, rel);
            for i in 0..6 { ws.c[child][i] -= propagated[i]; }
        }
        for joint in (0..self.joint_num).rev() {
            let child = self.child_link[joint]; let parent = self.parent_link[joint];
            let mut ia = ws.ia[child]; let mut pa = aba_add(ws.pa[child], mat6_vec6(ia, ws.c[child]));
            if self.q_index[joint] >= 0 {
                let u_vec = mat6_vec6(ia, ws.s[joint]); let d = aba_dot(ws.s[joint], u_vec);
                if !d.is_finite() || d <= 1e-12 { return Err("singular articulated inertia"); }
                let u = tau[self.q_index[joint] as usize] - aba_dot(ws.s[joint], ws.pa[child]);
                for row in 0..6 { for col in 0..6 { ia[row][col] -= u_vec[row] * u_vec[col] / d; }}
                pa = aba_add(aba_add(ws.pa[child], mat6_vec6(ia, ws.c[child])), aba_scale(u_vec, u / d));
                ws.u_vec[joint] = u_vec; ws.d[joint] = d; ws.u[joint] = u;
            }
            let rel = sub3(flat3(&ws.kinematics.p, child), flat3(&ws.kinematics.p, parent));
            let mut shifted = [[0.0; 6]; 6];
            for col in 0..6 {
                let mut e = [0.0; 6]; e[col] = 1.0;
                let value = aba_shift_force_to_parent(mat6_vec6(ia, aba_shift_motion_to_child(e, rel)), rel);
                for row in 0..6 { shifted[row][col] = value[row]; }
            }
            for row in 0..6 { for col in 0..6 { ws.ia[parent][row][col] += shifted[row][col]; }}
            ws.pa[parent] = aba_add(ws.pa[parent], aba_shift_force_to_parent(pa, rel));
        }
        ws.accel[0] = [0.0; 6];
        for joint in 0..self.joint_num {
            let parent = self.parent_link[joint]; let child = self.child_link[joint];
            let rel = sub3(flat3(&ws.kinematics.p, child), flat3(&ws.kinematics.p, parent));
            let mut accel = aba_add(aba_shift_motion_to_child(ws.accel[parent], rel), ws.c[child]);
            if self.q_index[joint] >= 0 {
                let qdd = (ws.u[joint] - aba_dot(ws.u_vec[joint], accel)) / ws.d[joint];
                ws.qdd[self.q_index[joint] as usize] = qdd;
                accel = aba_add(accel, aba_scale(ws.s[joint], qdd));
            }
            ws.accel[child] = accel;
        }
        Ok(())
    }

    /// Factor the fixed-base joint-space mass operator at `q`.
    ///
    /// The stored `U`/`D` and reduced articulated inertias are independent of
    /// velocity, gravity and effort.  They support many `M(q)^{-1} rhs`
    /// solves without rebuilding the articulated inertias.
    pub(crate) fn aba_factorize_mass_into(
        &self, q: &[f64], ws: &mut AbaWorkspace,
    ) -> Result<(), &'static str> {
        if q.len() != self.dof { return Err("q length must match robot dof"); }
        ws.zero.fill(0.0);
        self.forward_kinematics_into(q, &ws.zero, &ws.zero, &mut ws.kinematics);
        for link in 0..self.link_num {
            ws.ia[link] = aba_world_inertia(mat3_from_flat(&ws.kinematics.r, link), self.link_inertia[link]);
        }
        for joint in (0..self.joint_num).rev() {
            let parent = self.parent_link[joint]; let child = self.child_link[joint];
            let parent_r = mat3_from_flat(&ws.kinematics.r, parent);
            let axis = mat3_vec(mat3_mul(parent_r, self.origin_r[joint]), self.axis[joint]);
            ws.s[joint] = if self.is_prismatic[joint] {
                [0.0, 0.0, 0.0, axis[0], axis[1], axis[2]]
            } else { [axis[0], axis[1], axis[2], 0.0, 0.0, 0.0] };
            let mut ia = ws.ia[child];
            if self.q_index[joint] >= 0 {
                let u = mat6_vec6(ia, ws.s[joint]); let d = aba_dot(ws.s[joint], u);
                if !d.is_finite() || d <= 1e-12 { return Err("singular articulated inertia"); }
                for row in 0..6 { for col in 0..6 { ia[row][col] -= u[row] * u[col] / d; }}
                ws.u_vec[joint] = u; ws.d[joint] = d;
            }
            // Keep the reduced child inertia for the following triangular solve.
            ws.ia[child] = ia;
            let rel = sub3(flat3(&ws.kinematics.p, child), flat3(&ws.kinematics.p, parent));
            let mut shifted = [[0.0; 6]; 6];
            for col in 0..6 {
                let mut e = [0.0; 6]; e[col] = 1.0;
                let value = aba_shift_force_to_parent(mat6_vec6(ia, aba_shift_motion_to_child(e, rel)), rel);
                for row in 0..6 { shifted[row][col] = value[row]; }
            }
            for row in 0..6 { for col in 0..6 { ws.ia[parent][row][col] += shifted[row][col]; }}
        }
        Ok(())
    }

    /// Apply the cached articulated-body factor to one joint-space right hand side.
    pub(crate) fn aba_solve_mass_into(
        &self, rhs: &[f64], ws: &mut AbaWorkspace,
    ) -> Result<(), &'static str> {
        if rhs.len() != self.dof { return Err("rhs length must match robot dof"); }
        ws.pa.fill([0.0; 6]); ws.qdd.fill(0.0); ws.accel.fill([0.0; 6]);
        for joint in (0..self.joint_num).rev() {
            let parent = self.parent_link[joint]; let child = self.child_link[joint];
            let mut force = ws.pa[child];
            if self.q_index[joint] >= 0 {
                let u = rhs[self.q_index[joint] as usize] - aba_dot(ws.s[joint], force);
                ws.u[joint] = u;
                force = aba_add(force, aba_scale(ws.u_vec[joint], u / ws.d[joint]));
            }
            let rel = sub3(flat3(&ws.kinematics.p, child), flat3(&ws.kinematics.p, parent));
            ws.pa[parent] = aba_add(ws.pa[parent], aba_shift_force_to_parent(force, rel));
        }
        for joint in 0..self.joint_num {
            let parent = self.parent_link[joint]; let child = self.child_link[joint];
            let rel = sub3(flat3(&ws.kinematics.p, child), flat3(&ws.kinematics.p, parent));
            let mut accel = aba_shift_motion_to_child(ws.accel[parent], rel);
            if self.q_index[joint] >= 0 {
                let qdd = (ws.u[joint] - aba_dot(ws.u_vec[joint], accel)) / ws.d[joint];
                ws.qdd[self.q_index[joint] as usize] = qdd;
                accel = aba_add(accel, aba_scale(ws.s[joint], qdd));
            }
            ws.accel[child] = accel;
        }
        Ok(())
    }

    pub(crate) fn rnea_jacobian_into(
        &self,
        q: &[f64],
        v: &[f64],
        a: &[f64],
        gravity: [f64; 3],
        interleaved: bool,
    ) -> Vec<f64> {
        let interleaved_jac = self.rnea_jacobian_bulk_interleaved_into(q, v, a, gravity);
        if interleaved {
            return interleaved_jac;
        }
        let cols = 3 * self.dof;
        let mut grouped = vec![0.0; self.dof * cols];
        for row in 0..self.dof {
            for joint in 0..self.dof {
                grouped[row * cols + joint] = interleaved_jac[row * cols + 3 * joint];
                grouped[row * cols + self.dof + joint] =
                    interleaved_jac[row * cols + 3 * joint + 1];
                grouped[row * cols + 2 * self.dof + joint] =
                    interleaved_jac[row * cols + 3 * joint + 2];
            }
        }
        grouped
    }

    pub(crate) fn rnea_jacobian_bulk_interleaved_into(
        &self,
        q: &[f64],
        v: &[f64],
        a: &[f64],
        gravity: [f64; 3],
    ) -> Vec<f64> {
        let cols = 3 * self.dof;
        let mut base = Workspace::new(self);
        let mut deriv = BulkDerivativeWorkspace::new(self, cols);
        let mut out = vec![0.0; self.dof * cols];
        self.rnea_jacobian_bulk_interleaved_fill(
            q,
            v,
            a,
            gravity,
            &self.link_motion_columns,
            &mut base,
            &mut deriv,
            &mut out,
        );
        out
    }

    pub(crate) fn rnea_jacobian_bulk_interleaved_fill(
        &self,
        q: &[f64],
        v: &[f64],
        a: &[f64],
        gravity: [f64; 3],
        motion_cols: &[Vec<usize>],
        base: &mut Workspace,
        deriv: &mut BulkDerivativeWorkspace,
        out: &mut [f64],
    ) {
        self.rnea_with_gravity_into(q, v, a, gravity, base);
        deriv.clear();
        self.forward_kinematics_bulk_derivative_into(q, v, a, base, deriv, motion_cols);
        self.link_force_bulk_derivative_into(base, deriv, motion_cols, gravity);
        self.backward_force_bulk_derivative_into(base, deriv);
        out.copy_from_slice(&deriv.tau);
    }

    pub(crate) fn rnea_jacobian_matmul_interleaved_fill(
        &self,
        q: &[f64],
        v: &[f64],
        a: &[f64],
        gravity: [f64; 3],
        rhs: &[f64],
        rhs_cols: usize,
        base: &mut Workspace,
        deriv: &mut BulkDerivativeWorkspace,
        out: &mut [f64],
    ) {
        self.rnea_with_gravity_into(q, v, a, gravity, base);
        deriv.clear();
        self.forward_kinematics_directional_derivative_into(q, v, a, rhs, rhs_cols, base, deriv);
        let all_cols = all_directional_cols(self.link_num, rhs_cols);
        self.link_force_bulk_derivative_into(base, deriv, &all_cols, gravity);
        self.backward_force_directional_derivative_into(base, deriv);
        out.copy_from_slice(&deriv.tau);
    }

    pub(crate) fn forward_kinematics_directional_derivative_into(
        &self,
        q: &[f64],
        v: &[f64],
        a: &[f64],
        rhs: &[f64],
        rhs_cols: usize,
        base: &Workspace,
        deriv: &mut BulkDerivativeWorkspace,
    ) {
        let cols = deriv.cols;
        debug_assert_eq!(cols, rhs_cols);
        for j in 0..self.joint_num {
            let parent = self.parent_link[j];
            let child = self.child_link[j];
            let parent_r = mat3_from_flat(&base.r, parent);
            let parent_p = flat3(&base.p, parent);
            let parent_w = flat3(&base.w, parent);
            let parent_alpha = flat3(&base.alpha, parent);
            let joint_r0 = mat3_mul(parent_r, self.origin_r[j]);
            let joint_p = add3(parent_p, mat3_vec(parent_r, self.origin_p[j]));
            let rel = sub3(joint_p, parent_p);
            let qi = self.q_index[j];
            let rj = if qi >= 0 {
                rot_axis(self.axis[j], q[qi as usize])
            } else {
                eye3()
            };
            let drj_dq = if qi >= 0 {
                rot_axis_derivative(self.axis[j], q[qi as usize])
            } else {
                [[0.0; 3]; 3]
            };

            for col in 0..rhs_cols {
                let parent_dr = mat3_col(&deriv.r, parent, col, cols);
                let parent_dp = flat3_col(&deriv.p, parent, col, cols);
                let parent_dw = flat3_col(&deriv.w, parent, col, cols);
                let parent_dlin_v = flat3_col(&deriv.lin_v, parent, col, cols);
                let parent_dalpha = flat3_col(&deriv.alpha, parent, col, cols);
                let parent_dlin_a = flat3_col(&deriv.lin_a, parent, col, cols);
                let joint_dr0 = mat3_mul(parent_dr, self.origin_r[j]);
                let joint_dp = add3(parent_dp, mat3_vec(parent_dr, self.origin_p[j]));
                let drel = sub3(joint_dp, parent_dp);

                if qi >= 0 {
                    let qi = qi as usize;
                    let dq = rhs[(3 * qi) * rhs_cols + col];
                    let dv = rhs[(3 * qi + 1) * rhs_cols + col];
                    let da = rhs[(3 * qi + 2) * rhs_cols + col];
                    let axis_world = mat3_vec(joint_r0, self.axis[j]);
                    let daxis_world = mat3_vec(joint_dr0, self.axis[j]);
                    let drj = scale_mat3(drj_dq, dq);
                    set_mat3_col(
                        &mut deriv.r,
                        child,
                        col,
                        cols,
                        add_mat3(mat3_mul(joint_dr0, rj), mat3_mul(joint_r0, drj)),
                    );
                    set_flat3_col(&mut deriv.p, child, col, cols, joint_dp);
                    set_flat3_col(
                        &mut deriv.w,
                        child,
                        col,
                        cols,
                        add3(
                            parent_dw,
                            add3(scale3(daxis_world, v[qi]), scale3(axis_world, dv)),
                        ),
                    );
                    set_flat3_col(
                        &mut deriv.lin_v,
                        child,
                        col,
                        cols,
                        add3(
                            parent_dlin_v,
                            add3(cross(parent_dw, rel), cross(parent_w, drel)),
                        ),
                    );
                    let axis_v = scale3(axis_world, v[qi]);
                    let daxis_v = add3(scale3(daxis_world, v[qi]), scale3(axis_world, dv));
                    set_flat3_col(
                        &mut deriv.alpha,
                        child,
                        col,
                        cols,
                        add3(
                            add3(
                                parent_dalpha,
                                add3(scale3(daxis_world, a[qi]), scale3(axis_world, da)),
                            ),
                            add3(cross(parent_dw, axis_v), cross(parent_w, daxis_v)),
                        ),
                    );
                    let w_cross_rel = cross(parent_w, rel);
                    let dw_cross_rel = add3(cross(parent_dw, rel), cross(parent_w, drel));
                    set_flat3_col(
                        &mut deriv.lin_a,
                        child,
                        col,
                        cols,
                        add3(
                            add3(
                                parent_dlin_a,
                                add3(cross(parent_dalpha, rel), cross(parent_alpha, drel)),
                            ),
                            add3(cross(parent_dw, w_cross_rel), cross(parent_w, dw_cross_rel)),
                        ),
                    );
                } else {
                    set_mat3_col(&mut deriv.r, child, col, cols, joint_dr0);
                    set_flat3_col(&mut deriv.p, child, col, cols, joint_dp);
                    set_flat3_col(&mut deriv.w, child, col, cols, parent_dw);
                    set_flat3_col(
                        &mut deriv.lin_v,
                        child,
                        col,
                        cols,
                        add3(
                            parent_dlin_v,
                            add3(cross(parent_dw, rel), cross(parent_w, drel)),
                        ),
                    );
                    set_flat3_col(&mut deriv.alpha, child, col, cols, parent_dalpha);
                    let w_cross_rel = cross(parent_w, rel);
                    let dw_cross_rel = add3(cross(parent_dw, rel), cross(parent_w, drel));
                    set_flat3_col(
                        &mut deriv.lin_a,
                        child,
                        col,
                        cols,
                        add3(
                            add3(
                                parent_dlin_a,
                                add3(cross(parent_dalpha, rel), cross(parent_alpha, drel)),
                            ),
                            add3(cross(parent_dw, w_cross_rel), cross(parent_w, dw_cross_rel)),
                        ),
                    );
                }
            }
        }
    }

    pub(crate) fn forward_kinematics_bulk_derivative_into(
        &self,
        q: &[f64],
        v: &[f64],
        a: &[f64],
        base: &Workspace,
        deriv: &mut BulkDerivativeWorkspace,
        motion_cols: &[Vec<usize>],
    ) {
        let cols = deriv.cols;
        for j in 0..self.joint_num {
            let parent = self.parent_link[j];
            let child = self.child_link[j];
            let parent_r = mat3_from_flat(&base.r, parent);
            let parent_p = flat3(&base.p, parent);
            let parent_w = flat3(&base.w, parent);
            let parent_alpha = flat3(&base.alpha, parent);
            let joint_r0 = mat3_mul(parent_r, self.origin_r[j]);
            let joint_p = add3(parent_p, mat3_vec(parent_r, self.origin_p[j]));
            let rel = sub3(joint_p, parent_p);
            let qi = self.q_index[j];
            let rj = if qi >= 0 {
                rot_axis(self.axis[j], q[qi as usize])
            } else {
                eye3()
            };
            let drj_dq = if qi >= 0 {
                rot_axis_derivative(self.axis[j], q[qi as usize])
            } else {
                [[0.0; 3]; 3]
            };

            for &col in &motion_cols[child] {
                let parent_dr = mat3_col(&deriv.r, parent, col, cols);
                let parent_dp = flat3_col(&deriv.p, parent, col, cols);
                let parent_dw = flat3_col(&deriv.w, parent, col, cols);
                let parent_dlin_v = flat3_col(&deriv.lin_v, parent, col, cols);
                let parent_dalpha = flat3_col(&deriv.alpha, parent, col, cols);
                let parent_dlin_a = flat3_col(&deriv.lin_a, parent, col, cols);
                let joint_dr0 = mat3_mul(parent_dr, self.origin_r[j]);
                let joint_dp = add3(parent_dp, mat3_vec(parent_dr, self.origin_p[j]));
                let drel = sub3(joint_dp, parent_dp);

                if qi >= 0 {
                    let qi = qi as usize;
                    let dq = if col == 3 * qi { 1.0 } else { 0.0 };
                    let dv = if col == 3 * qi + 1 { 1.0 } else { 0.0 };
                    let da = if col == 3 * qi + 2 { 1.0 } else { 0.0 };
                    let axis_world = mat3_vec(joint_r0, self.axis[j]);
                    let daxis_world = mat3_vec(joint_dr0, self.axis[j]);
                    let drj = scale_mat3(drj_dq, dq);
                    set_mat3_col(
                        &mut deriv.r,
                        child,
                        col,
                        cols,
                        add_mat3(mat3_mul(joint_dr0, rj), mat3_mul(joint_r0, drj)),
                    );
                    set_flat3_col(&mut deriv.p, child, col, cols, joint_dp);
                    set_flat3_col(
                        &mut deriv.w,
                        child,
                        col,
                        cols,
                        add3(
                            parent_dw,
                            add3(scale3(daxis_world, v[qi]), scale3(axis_world, dv)),
                        ),
                    );
                    set_flat3_col(
                        &mut deriv.lin_v,
                        child,
                        col,
                        cols,
                        add3(
                            parent_dlin_v,
                            add3(cross(parent_dw, rel), cross(parent_w, drel)),
                        ),
                    );
                    let axis_v = scale3(axis_world, v[qi]);
                    let daxis_v = add3(scale3(daxis_world, v[qi]), scale3(axis_world, dv));
                    set_flat3_col(
                        &mut deriv.alpha,
                        child,
                        col,
                        cols,
                        add3(
                            add3(
                                parent_dalpha,
                                add3(scale3(daxis_world, a[qi]), scale3(axis_world, da)),
                            ),
                            add3(cross(parent_dw, axis_v), cross(parent_w, daxis_v)),
                        ),
                    );
                    let w_cross_rel = cross(parent_w, rel);
                    let dw_cross_rel = add3(cross(parent_dw, rel), cross(parent_w, drel));
                    set_flat3_col(
                        &mut deriv.lin_a,
                        child,
                        col,
                        cols,
                        add3(
                            add3(
                                parent_dlin_a,
                                add3(cross(parent_dalpha, rel), cross(parent_alpha, drel)),
                            ),
                            add3(cross(parent_dw, w_cross_rel), cross(parent_w, dw_cross_rel)),
                        ),
                    );
                } else {
                    set_mat3_col(&mut deriv.r, child, col, cols, joint_dr0);
                    set_flat3_col(&mut deriv.p, child, col, cols, joint_dp);
                    set_flat3_col(&mut deriv.w, child, col, cols, parent_dw);
                    set_flat3_col(
                        &mut deriv.lin_v,
                        child,
                        col,
                        cols,
                        add3(
                            parent_dlin_v,
                            add3(cross(parent_dw, rel), cross(parent_w, drel)),
                        ),
                    );
                    set_flat3_col(&mut deriv.alpha, child, col, cols, parent_dalpha);
                    let w_cross_rel = cross(parent_w, rel);
                    let dw_cross_rel = add3(cross(parent_dw, rel), cross(parent_w, drel));
                    set_flat3_col(
                        &mut deriv.lin_a,
                        child,
                        col,
                        cols,
                        add3(
                            add3(
                                parent_dlin_a,
                                add3(cross(parent_dalpha, rel), cross(parent_alpha, drel)),
                            ),
                            add3(cross(parent_dw, w_cross_rel), cross(parent_w, dw_cross_rel)),
                        ),
                    );
                }
            }
        }
    }

    pub(crate) fn link_force_bulk_derivative_into(
        &self,
        base: &Workspace,
        deriv: &mut BulkDerivativeWorkspace,
        motion_cols: &[Vec<usize>],
        gravity: [f64; 3],
    ) {
        let cols = deriv.cols;
        for link_id in 1..self.link_num {
            let r = mat3_from_flat(&base.r, link_id);
            let rt = mat3_transpose(r);
            let v_world = [flat3(&base.w, link_id), flat3(&base.lin_v, link_id)];
            let a_world = [flat3(&base.alpha, link_id), flat3(&base.lin_a, link_id)];
            let v_local = [mat3_vec(rt, v_world[0]), mat3_vec(rt, v_world[1])];
            let a_local_ang = mat3_vec(rt, a_world[0]);
            let a_local_lin = sub3(
                sub3(mat3_vec(rt, a_world[1]), mat3_vec(rt, gravity)),
                cross(v_local[0], v_local[1]),
            );
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

            for &col in &motion_cols[link_id] {
                let dr = mat3_col(&deriv.r, link_id, col, cols);
                let drt = mat3_transpose(dr);
                let dv_world = [
                    flat3_col(&deriv.w, link_id, col, cols),
                    flat3_col(&deriv.lin_v, link_id, col, cols),
                ];
                let da_world = [
                    flat3_col(&deriv.alpha, link_id, col, cols),
                    flat3_col(&deriv.lin_a, link_id, col, cols),
                ];
                let dv_local = [
                    add3(mat3_vec(drt, v_world[0]), mat3_vec(rt, dv_world[0])),
                    add3(mat3_vec(drt, v_world[1]), mat3_vec(rt, dv_world[1])),
                ];
                let da_local_ang = add3(mat3_vec(drt, a_world[0]), mat3_vec(rt, da_world[0]));
                let da_local_lin = sub3(
                    sub3(
                        add3(mat3_vec(drt, a_world[1]), mat3_vec(rt, da_world[1])),
                        mat3_vec(drt, gravity),
                    ),
                    add3(
                        cross(dv_local[0], v_local[1]),
                        cross(v_local[0], dv_local[1]),
                    ),
                );
                let da_local = [da_local_ang, da_local_lin];
                let dmomentum = mat6_vec(self.link_inertia[link_id], dv_local);
                let dinertial = mat6_vec(self.link_inertia[link_id], da_local);
                let dforce_local = [
                    add3(
                        dinertial[0],
                        add3(
                            add3(
                                cross(dv_local[0], momentum[0]),
                                cross(v_local[0], dmomentum[0]),
                            ),
                            add3(
                                cross(dv_local[1], momentum[1]),
                                cross(v_local[1], dmomentum[1]),
                            ),
                        ),
                    ),
                    add3(
                        dinertial[1],
                        add3(
                            cross(dv_local[0], momentum[1]),
                            cross(v_local[0], dmomentum[1]),
                        ),
                    ),
                ];
                set_force_col(
                    &mut deriv.forces,
                    link_id,
                    col,
                    cols,
                    add3(mat3_vec(dr, force_local[0]), mat3_vec(r, dforce_local[0])),
                    add3(mat3_vec(dr, force_local[1]), mat3_vec(r, dforce_local[1])),
                );
            }
        }
    }

    pub(crate) fn backward_force_bulk_derivative_into(
        &self,
        base: &Workspace,
        deriv: &mut BulkDerivativeWorkspace,
    ) {
        let cols = deriv.cols;
        for j in (0..self.joint_num).rev() {
            let parent = self.parent_link[j];
            let child = self.child_link[j];
            let qi = self.q_index[j];
            let parent_r = mat3_from_flat(&base.r, parent);
            let joint_r = mat3_mul(parent_r, self.origin_r[j]);
            let axis_world = mat3_vec(joint_r, self.axis[j]);
            let rel = sub3(flat3(&base.p, child), flat3(&base.p, parent));
            let child_force = force_force(&base.forces, child);
            let child_torque = force_torque(&base.forces, child);
            let child_cols = &self.link_subtree_motion_columns[child];

            for col in child_cols.iter().copied() {
                if qi >= 0 {
                    let parent_dr = mat3_col(&deriv.r, parent, col, cols);
                    let joint_dr = mat3_mul(parent_dr, self.origin_r[j]);
                    let daxis_world = mat3_vec(joint_dr, self.axis[j]);
                    let dchild_torque = force_torque_col(&deriv.forces, child, col, cols);
                    set_tau_col(
                        &mut deriv.tau,
                        qi as usize,
                        col,
                        cols,
                        dot3(daxis_world, child_torque) + dot3(axis_world, dchild_torque),
                    );
                }
                let drel = sub3(
                    flat3_col(&deriv.p, child, col, cols),
                    flat3_col(&deriv.p, parent, col, cols),
                );
                let dchild_torque = force_torque_col(&deriv.forces, child, col, cols);
                let dchild_force = force_force_col(&deriv.forces, child, col, cols);
                add_shifted_force_parent_derivative_col(
                    &mut deriv.forces,
                    parent,
                    col,
                    cols,
                    rel,
                    drel,
                    child_force,
                    dchild_torque,
                    dchild_force,
                );
            }
        }
    }

    pub(crate) fn backward_force_directional_derivative_into(
        &self,
        base: &Workspace,
        deriv: &mut BulkDerivativeWorkspace,
    ) {
        let cols = deriv.cols;
        for j in (0..self.joint_num).rev() {
            let parent = self.parent_link[j];
            let child = self.child_link[j];
            let qi = self.q_index[j];
            let parent_r = mat3_from_flat(&base.r, parent);
            let joint_r = mat3_mul(parent_r, self.origin_r[j]);
            let axis_world = mat3_vec(joint_r, self.axis[j]);
            let rel = sub3(flat3(&base.p, child), flat3(&base.p, parent));
            let child_force = force_force(&base.forces, child);
            let child_torque = force_torque(&base.forces, child);

            for col in 0..cols {
                if qi >= 0 {
                    let parent_dr = mat3_col(&deriv.r, parent, col, cols);
                    let joint_dr = mat3_mul(parent_dr, self.origin_r[j]);
                    let daxis_world = mat3_vec(joint_dr, self.axis[j]);
                    let dchild_torque = force_torque_col(&deriv.forces, child, col, cols);
                    set_tau_col(
                        &mut deriv.tau,
                        qi as usize,
                        col,
                        cols,
                        dot3(daxis_world, child_torque) + dot3(axis_world, dchild_torque),
                    );
                }
                let drel = sub3(
                    flat3_col(&deriv.p, child, col, cols),
                    flat3_col(&deriv.p, parent, col, cols),
                );
                let dchild_torque = force_torque_col(&deriv.forces, child, col, cols);
                let dchild_force = force_force_col(&deriv.forces, child, col, cols);
                add_shifted_force_parent_derivative_col(
                    &mut deriv.forces,
                    parent,
                    col,
                    cols,
                    rel,
                    drel,
                    child_force,
                    dchild_torque,
                    dchild_force,
                );
            }
        }
    }
}

fn all_directional_cols(link_num: usize, cols: usize) -> Vec<Vec<usize>> {
    let one: Vec<usize> = (0..cols).collect();
    vec![one; link_num]
}
