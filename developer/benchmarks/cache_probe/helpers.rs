impl RustCompiledRobot {
    // Fill only fields omitted by dynamics_cmtm_order1_cached_into. Reuse the
    // already computed link kinematics and subtree joint momentum.
    fn probe_complete_order1(&self, motion: &[f64], ws: &mut DynamicsCmtmWorkspace) {
        fill_factorial_table(&mut ws.factorial);
        fill_factorial_table(&mut ws.cmtm.factorial);
        for j in 0..self.joint_num {
            let qi = self.q_index[j];
            if qi < 0 {
                set_mat4(&mut ws.cmtm.joint_mat, j, mat4_from_rot_pos(eye3(), [0.0; 3]));
                for k in 0..2 { set_vec6_flat(&mut ws.cmtm.joint_vecs, j * 2 + k, [0.0; 6]); }
            } else {
                let start = qi as usize * 3;
                set_mat4(&mut ws.cmtm.joint_mat, j, mat4_from_rot_pos(rot_axis(self.axis[j], motion[start]), [0.0; 3]));
                for k in 0..2 {
                    let a = scale3(self.axis[j], motion[start + k + 1]);
                    set_vec6_flat(&mut ws.cmtm.joint_vecs, j * 2 + k, [a[0], a[1], a[2], 0.0, 0.0, 0.0]);
                }
            }
            let vel = cmtm_vecs_slice(&ws.cmtm.link_vecs, self.child_link[j], 3);
            let momentum = cmvec_slice(&ws.joint_momentum, j, 2);
            let force = add6(vec6_from_flat(momentum, 1), hat_adj_wrench_vec6(vec6_from_flat(vel, 0), vec6_from_flat(momentum, 0)));
            set_vec6_flat(&mut ws.joint_force, j, force);
        }
        for link in 0..self.link_num {
            let vel = cmtm_vecs_slice(&ws.cmtm.link_vecs, link, 3);
            let v = vec6_from_flat(vel, 0);
            let m = mat6_vec6(self.link_inertia[link], v);
            let dm = mat6_vec6(self.link_inertia[link], vec6_from_flat(vel, 1));
            set_vec6_flat(&mut ws.link_momentum, link * 2, m);
            set_vec6_flat(&mut ws.link_momentum, link * 2 + 1, dm);
            set_vec6_flat(&mut ws.link_force, link, add6(dm, hat_adj_wrench_vec6(v, m)));
        }
    }
}
