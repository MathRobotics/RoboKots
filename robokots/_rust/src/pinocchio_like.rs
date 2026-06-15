use crate::spatial::*;
use crate::types::RustCompiledRobot;

pub(crate) struct PinocchioLikeWorkspace {
    pub(crate) r: Vec<f64>,
    pub(crate) p: Vec<f64>,
    pub(crate) w: Vec<f64>,
    pub(crate) lin_v: Vec<f64>,
    pub(crate) alpha: Vec<f64>,
    pub(crate) lin_a: Vec<f64>,
    pub(crate) forces: Vec<f64>,
    pub(crate) tau: Vec<f64>,
    pub(crate) jac: Vec<f64>,
    active_axes: Vec<[f64; 3]>,
    active_points: Vec<[f64; 3]>,
    zero_motion: Vec<f64>,
}

impl PinocchioLikeWorkspace {
    pub(crate) fn new(robot: &RustCompiledRobot) -> Self {
        Self {
            r: vec![0.0; robot.link_num * 9],
            p: vec![0.0; robot.link_num * 3],
            w: vec![0.0; robot.link_num * 3],
            lin_v: vec![0.0; robot.link_num * 3],
            alpha: vec![0.0; robot.link_num * 3],
            lin_a: vec![0.0; robot.link_num * 3],
            forces: vec![0.0; robot.link_num * 6],
            tau: vec![0.0; robot.dof],
            jac: vec![0.0; robot.link_num * 6 * robot.dof],
            active_axes: vec![[0.0; 3]; robot.dof],
            active_points: vec![[0.0; 3]; robot.dof],
            zero_motion: vec![0.0; robot.dof],
        }
    }

    fn clear_kinematics(&mut self) {
        self.r.fill(0.0);
        self.p.fill(0.0);
        self.w.fill(0.0);
        self.lin_v.fill(0.0);
        self.alpha.fill(0.0);
        self.lin_a.fill(0.0);
    }
}

impl RustCompiledRobot {
    pub(crate) fn pinocchio_forward_kinematics_into(
        &self,
        q: &[f64],
        v: &[f64],
        a: &[f64],
        ws: &mut PinocchioLikeWorkspace,
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
            let rel = sub3(joint_p, parent_p);
            let qi = self.q_index[j];

            if qi >= 0 {
                let qi = qi as usize;
                let axis_world = mat3_vec(joint_r0, self.axis[j]);
                let rj = rot_axis(self.axis[j], q[qi]);
                set_mat3(&mut ws.r, child, mat3_mul(joint_r0, rj));
                set_flat3(&mut ws.p, child, joint_p);
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

    pub(crate) fn pinocchio_rnea_into(
        &self,
        q: &[f64],
        v: &[f64],
        a: &[f64],
        ws: &mut PinocchioLikeWorkspace,
    ) {
        self.pinocchio_forward_kinematics_into(q, v, a, ws);
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

    pub(crate) fn pinocchio_joint_jacobians_into(
        &self,
        q: &[f64],
        ws: &mut PinocchioLikeWorkspace,
    ) {
        let mut zero = std::mem::take(&mut ws.zero_motion);
        if zero.len() != self.dof {
            zero.resize(self.dof, 0.0);
        }
        zero.fill(0.0);
        self.pinocchio_forward_kinematics_into(q, &zero, &zero, ws);
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
            let parent_r = mat3_from_flat(&ws.r, parent);
            let joint_r = mat3_mul(parent_r, self.origin_r[j]);
            ws.active_axes[qi] = mat3_vec(joint_r, self.axis[j]);
            ws.active_points[qi] = add3(flat3(&ws.p, parent), mat3_vec(parent_r, self.origin_p[j]));
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
}
