//! Dense motion-coordinate derivatives for rigid fixed/revolute trees.
//!
//! Assemble kinematic derivative columns directly, using ancestor sparsity and
//! time-order causality, then share the analytic momentum/force recurrence and
//! output projections with the product kernels. No identity RHS or JVP call is
//! involved. Input columns are owner-major: q, qdot, qddot, ... per active joint.
use crate::cmtm_generic::{set_tangent_mat4, tangent_mat4};
use crate::dynamics_outputs::DynamicsOutput;
use crate::spatial::*;
use crate::types::RustCompiledRobot;
use crate::workspace::{CmtmWorkspace, DynamicsCmtmTangentWorkspace, DynamicsCmtmWorkspace};

type Block = [[f64; 6]; 6];

fn derivative_vector(data: &[f64], link: usize, time: usize, count: usize,
                     cols: usize, col: usize) -> [f64; 6] {
    let mut value = [0.0; 6];
    for c in 0..6 { value[c] = data[((link * count + time) * 6 + c) * cols + col]; }
    value
}

fn put_derivative(data: &mut [f64], link: usize, time: usize, count: usize,
                  cols: usize, col: usize, value: [f64; 6]) {
    for c in 0..6 { data[((link * count + time) * 6 + c) * cols + col] = value[c]; }
}

impl RustCompiledRobot {
    pub(crate) fn dynamics_jacobian_from_state_into(
        &self, motion: &[f64], outputs: &[DynamicsOutput], order: usize,
        gravity: [f64; 3], primal: &DynamicsCmtmWorkspace,
        derivatives: &mut DynamicsCmtmTangentWorkspace, out: &mut [f64],
    ) {
        debug_assert!(order >= 2);
        debug_assert_eq!(derivatives.rhs_cols, self.dof * order);
        self.kinematics_coordinate_derivatives_into(motion, order, &primal.cmtm, derivatives);
        self.dynamics_derivatives_from_kinematics_into(order - 2, gravity, primal, derivatives);
        self.selected_derivatives_into(outputs, order, primal, derivatives, out);
    }

    fn kinematics_coordinate_derivatives_into(
        &self, motion: &[f64], order: usize, primal: &CmtmWorkspace,
        derivatives: &mut DynamicsCmtmTangentWorkspace,
    ) {
        let cols = self.dof * order;
        let count = order - 1;
        let fact = &primal.factorial;
        derivatives.clear();
        // Per-edge transform coefficients are reused across all ancestor columns.
        let mut x = vec![[[0.0; 6]; 6]; count];
        let mut dx: Vec<Vec<Block>> = vec![vec![[[0.0; 6]; 6]; count]; count];
        for j in 0..self.joint_num {
            let parent = self.parent_link[j];
            let child = self.child_link[j];
            let origin = mat4_from_rot_pos(self.origin_r[j], self.origin_p[j]);
            let joint = mat4_from_flat(&primal.joint_mat, j);
            let relative = mat4_mul(origin, joint);
            let parent_mat = mat4_from_flat(&primal.link_mat, parent);
            let velocities = cmtm_vecs_slice(&primal.link_vecs, parent, order);
            let joint_velocities = cmtm_vecs_slice(&primal.joint_vecs, j, order);
            let axis = [self.axis[j][0], self.axis[j][1], self.axis[j][2], 0.0, 0.0, 0.0];
            let qi = self.q_index[j];

            // X is the inverse motion adjoint. The spatial helper produces
            // factorial-normalized wrench blocks; swap angular/linear halves.
            cmtm_mat_inv_adj_wrench_blocks_into(relative, joint_velocities, count, fact, &mut x);
            for block in &mut x {
                let wrench = *block;
                for r in 0..6 { for c in 0..6 { block[r][c] = wrench[(r+3)%6][(c+3)%6]; }}
            }

            // Ancestor pose columns: d(T_parent T_relative)=dT_parent T_relative.
            for &ancestor in &self.link_ancestors[parent] {
                let col = ancestor * order;
                let dparent = tangent_mat4(&derivatives.link_mat, parent, cols, col);
                set_tangent_mat4(&mut derivatives.link_mat, child, cols, col, mat4_mul(dparent, relative));
            }

            // V_child^(k) = sum_t k!/(k-t)! X[t] V_parent^(k-t) + S q^(k+1).
            // Parent columns have no derivative of this edge's transform.
            for &ancestor in &self.link_ancestors[parent] {
                for k in 0..count { for r in 0..=k+1 {
                    let col = ancestor * order + r;
                    let mut value = [0.0; 6];
                    for t in 0..=k {
                        if r > k - t + 1 { continue; }
                        let dv = derivative_vector(&derivatives.link_vecs, parent, k-t, count, cols, col);
                        value = add6(value, scale6(mat6_vec6(x[t], dv), fact[k]/fact[k-t]));
                    }
                    put_derivative(&mut derivatives.link_vecs, child, k, count, cols, col, value);
                }}
            }
            if qi < 0 { continue; }
            let start = qi as usize * order;
            let rotation_derivative = rot_axis_derivative(self.axis[j], motion[start]);
            let mut djoint = [[0.0; 4]; 4];
            for r in 0..3 { for c in 0..3 { djoint[r][c] = rotation_derivative[r][c]; }}
            set_tangent_mat4(&mut derivatives.joint_mat, j, cols, start, djoint);
            set_tangent_mat4(&mut derivatives.link_mat, child, cols, start,
                            mat4_mul(parent_mat, mat4_mul(origin, djoint)));
            for k in 0..count {
                put_derivative(&mut derivatives.joint_vecs, j, k, count, cols, start+k+1, axis);
            }

            // Differentiate Xdot=-ad(S qdot) X only with respect to this
            // joint's coordinates. For a revolute screw ad(S) has equal
            // angular/linear diagonal blocks, like its wrench counterpart.
            let ad = hat_adj_wrench(axis);
            for row in &mut dx { row.fill([[0.0; 6]; 6]); }
            dx[0][0] = scale_mat6(mat6_mul(ad, x[0]), -1.0);
            for k in 1..count { for r in 0..=k {
                let mut value = [[0.0; 6]; 6];
                for i in 0..k {
                    let prev = k-i-1;
                    if r <= prev {
                        value = add_mat6(value, scale_mat6(mat6_mul(ad, dx[prev][r]), motion[start+i+1]/fact[i]));
                    }
                    if r == i+1 {
                        value = add_mat6(value, scale_mat6(mat6_mul(ad, x[prev]), 1.0/fact[i]));
                    }
                }
                dx[k][r] = scale_mat6(value, -1.0/k as f64);
            }}
            for k in 0..count {
                for r in 0..=k {
                    let mut value = [0.0; 6];
                    for t in r..=k {
                        value = add6(value, scale6(mat6_vec6(dx[t][r], vec6_from_flat(velocities, k-t)), fact[k]/fact[k-t]));
                    }
                    put_derivative(&mut derivatives.link_vecs, child, k, count, cols, start+r, value);
                }
                put_derivative(&mut derivatives.link_vecs, child, k, count, cols, start+k+1, axis);
            }
        }
    }
}
