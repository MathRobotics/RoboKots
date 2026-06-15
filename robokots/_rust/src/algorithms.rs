use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::spatial::*;
use crate::types::RustCompiledRobot;
use crate::workspace::{BulkDerivativeWorkspace, Workspace};

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
                ws.tau[qi as usize] = dot3(axis_world, force_torque(&ws.forces, child));
            }
            let rel = sub3(flat3(&ws.p, child), flat3(&ws.p, parent));
            add_shifted_force_parent(&mut ws.forces, parent, child, rel);
        }
    }

    pub(crate) fn rnea_jacobian_into(
        &self,
        q: &[f64],
        v: &[f64],
        a: &[f64],
        interleaved: bool,
    ) -> Vec<f64> {
        let interleaved_jac = self.rnea_jacobian_bulk_interleaved_into(q, v, a);
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
    ) -> Vec<f64> {
        let cols = 3 * self.dof;
        let mut base = Workspace::new(self);
        let mut deriv = BulkDerivativeWorkspace::new(self, cols);
        let mut out = vec![0.0; self.dof * cols];
        self.rnea_jacobian_bulk_interleaved_fill(
            q,
            v,
            a,
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
        motion_cols: &[Vec<usize>],
        base: &mut Workspace,
        deriv: &mut BulkDerivativeWorkspace,
        out: &mut [f64],
    ) {
        self.rnea_into(q, v, a, base);
        deriv.clear();
        self.forward_kinematics_bulk_derivative_into(q, v, a, base, deriv, motion_cols);
        self.link_force_bulk_derivative_into(base, deriv, motion_cols);
        self.backward_force_bulk_derivative_into(base, deriv);
        out.copy_from_slice(&deriv.tau);
    }

    pub(crate) fn rnea_jacobian_matmul_interleaved_fill(
        &self,
        q: &[f64],
        v: &[f64],
        a: &[f64],
        rhs: &[f64],
        rhs_cols: usize,
        base: &mut Workspace,
        deriv: &mut BulkDerivativeWorkspace,
        out: &mut [f64],
    ) {
        self.rnea_into(q, v, a, base);
        deriv.clear();
        self.forward_kinematics_directional_derivative_into(q, v, a, rhs, rhs_cols, base, deriv);
        let all_cols = all_directional_cols(self.link_num, rhs_cols);
        self.link_force_bulk_derivative_into(base, deriv, &all_cols);
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
    ) {
        let cols = deriv.cols;
        for link_id in 1..self.link_num {
            let r = mat3_from_flat(&base.r, link_id);
            let rt = mat3_transpose(r);
            let v_world = [flat3(&base.w, link_id), flat3(&base.lin_v, link_id)];
            let a_world = [flat3(&base.alpha, link_id), flat3(&base.lin_a, link_id)];
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
                    add3(mat3_vec(drt, a_world[1]), mat3_vec(rt, da_world[1])),
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
