impl RustCompiledRobot {
    /// Fixed/revolute-only prototype. Coefficients of R_rel^T are factorial
    /// scaled, while stored spatial velocities remain ordinary derivatives.
    pub(crate) fn probe_spatial_kinematics_into(&self, motion: &[f64], order: usize, ws: &mut CmtmWorkspace) {
        ws.clear();
        fill_factorial_table(&mut ws.factorial);
        set_mat4(&mut ws.link_mat, 0, eye4());
        let count = order - 1;
        let mut rotations = vec![[[0.0; 3]; 3]; count];
        for j in 0..self.joint_num {
            let parent = self.parent_link[j];
            let child = self.child_link[j];
            let qi = self.q_index[j];
            let start = if qi >= 0 { qi as usize * order } else { 0 };
            let local_r = if qi >= 0 { rot_axis(self.axis[j], motion[start]) } else { eye3() };
            let local_mat = mat4_from_rot_pos(local_r, [0.0; 3]);
            let rel_r = mat3_mul(self.origin_r[j], local_r);
            let rel_mat = mat4_from_rot_pos(rel_r, self.origin_p[j]);
            let parent_mat = mat4_from_flat(&ws.link_mat, parent);
            set_mat4(&mut ws.link_mat, child, mat4_mul(parent_mat, rel_mat));
            set_mat4(&mut ws.joint_mat, j, local_mat);
            rotations[0] = mat3_transpose(rel_r);
            // dRinv/dt = -hat(axis*qdot) Rinv. Use normalized coefficients.
            if qi >= 0 {
                for n in 1..count {
                    let mut value = [[0.0; 3]; 3];
                    for k in 0..n {
                        let omega = scale3(self.axis[j], motion[start + k + 1] / ws.factorial[k]);
                        // hat(omega) A = -(A^T hat(omega))^T.
                        let term = mat3_transpose(mat3_mul_skew_right(mat3_transpose(rotations[n - 1 - k]), omega));
                        value = add_mat3(value, term);
                    }
                    rotations[n] = scale_mat3(value, 1.0 / n as f64);
                }
            }
            for n in 0..count {
                let mut omega = [0.0; 3];
                let mut linear = [0.0; 3];
                let terms = if qi >= 0 { n + 1 } else { 1 };
                for k in 0..terms {
                    let parent_v = vec6_from_flat(cmtm_vecs_slice(&ws.link_vecs, parent, order), n-k);
                    let w = [parent_v[0], parent_v[1], parent_v[2]];
                    let v = sub3([parent_v[3], parent_v[4], parent_v[5]], cross(self.origin_p[j], w));
                    let scale = ws.factorial[n] / ws.factorial[n-k];
                    omega = add3(omega, scale3(mat3_vec(rotations[k], w), scale));
                    linear = add3(linear, scale3(mat3_vec(rotations[k], v), scale));
                }
                let relative = if qi >= 0 { scale3(self.axis[j], motion[start+n+1]) } else { [0.0; 3] };
                omega = add3(omega, relative);
                set_vec6_flat(&mut ws.link_vecs, child * count + n,
                    [omega[0],omega[1],omega[2],linear[0],linear[1],linear[2]]);
                set_vec6_flat(&mut ws.joint_vecs, j * count + n,
                    [relative[0],relative[1],relative[2],0.0,0.0,0.0]);
            }
        }
    }
}
