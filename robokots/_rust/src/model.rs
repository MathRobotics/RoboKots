//! Native model input and compilation. No Python objects are stored or required.
use crate::error::{CoreResult, Error};
use crate::spatial::*;
use crate::types::RustCompiledRobot;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointKind {
    Fixed,
    Revolute,
    Prismatic,
}

impl JointKind {
    pub(crate) fn parse(kind: &str, representation: &str) -> CoreResult<Self> {
        match kind {
            "fixed" => Ok(Self::Fixed),
            "revolute" => Ok(Self::Revolute),
            "prismatic" => Ok(Self::Prismatic),
            "spherical" | "floating" => {
                let required = if kind == "spherical" { "rotation_vector" } else { "expmap" };
                if representation != required {
                    return Err(Error::new(format!("{kind} joints require q_representation='{required}'")));
                }
                Err(Error::new("Rust backend currently supports fixed/revolute joints only; spherical/floating joints are supported by the Python backend"))
            }
            _ => Err(Error::new("Rust RNEA supports fixed/revolute/prismatic joints only; use the Python backend for multi-DoF joints")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LinkModel {
    pub name: String,
    pub mass: f64,
    pub cog: [f64; 3],
    /// ixx, iyy, izz, ixy, ixz, iyz, about the center of gravity.
    pub inertia: [f64; 6],
}

#[derive(Clone, Debug)]
pub struct JointModel {
    pub name: String,
    pub parent_link: usize,
    pub child_link: usize,
    pub kind: JointKind,
    pub axis: [f64; 3],
    pub position: [f64; 3],
    /// Quaternion in w, x, y, z order.
    pub orientation: [f64; 4],
}

/// Links are indexed from zero, with link zero as the root.
/// Joints must be in parent-before-child order.
#[derive(Clone, Debug)]
pub struct RobotModel {
    pub links: Vec<LinkModel>,
    pub joints: Vec<JointModel>,
}

/// Read-only metadata for resolving named state requests without a Python model.
#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub dof: usize,
    pub link_names: Vec<String>,
    pub joints: Vec<JointInfo>,
    /// CMTM currently supports fixed/revolute rigid-link models.
    pub supports_cmtm: bool,
}

#[derive(Clone, Debug)]
pub struct JointInfo {
    pub id: usize,
    pub name: String,
    pub kind: JointKind,
    pub dof: usize,
    pub dof_index: usize,
    pub parent_link: usize,
    pub child_link: usize,
}

impl JointKind {
    pub fn as_str(self) -> &'static str {
        match self { Self::Fixed => "fixed", Self::Revolute => "revolute", Self::Prismatic => "prismatic" }
    }
}

impl RustCompiledRobot {
    pub fn model_info(&self) -> &ModelInfo { &self.info }

    pub(crate) fn check_cmtm_supported(&self) -> CoreResult<()> {
        if !self.info.supports_cmtm {
            return Err(Error::new("Rust CMTM supports fixed/revolute joints only"));
        }
        Ok(())
    }

    pub fn from_model(model: &RobotModel, allow_prismatic: bool) -> CoreResult<Self> {
        let link_num = model.links.len();
        let joint_num = model.joints.len();
        if link_num == 0 {
            return Err(Error::new("model must contain at least one link"));
        }
        for names in [model.links.iter().map(|x| x.name.as_str()).collect::<Vec<_>>(),
                      model.joints.iter().map(|x| x.name.as_str()).collect::<Vec<_>>()] {
            let mut seen = std::collections::HashSet::new();
            if names.iter().any(|name| name.is_empty() || !seen.insert(*name)) {
                return Err(Error::new("model owner names must be nonempty and unique within each owner type"));
            }
        }
        for link in &model.links {
            if !link.mass.is_finite() || link.mass < 0.0
                || !link.cog.iter().chain(&link.inertia).all(|x| x.is_finite()) {
                return Err(Error::new("link mass must be nonnegative and all inertial values finite"));
            }
        }
        for joint in &model.joints {
            let norm2 = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
            let qnorm = norm2(&joint.orientation);
            let anorm = norm2(&joint.axis);
            if !joint.position.iter().chain(&joint.orientation).chain(&joint.axis).all(|x| x.is_finite())
                || !qnorm.is_finite() || qnorm <= 0.0
                || (joint.kind != JointKind::Fixed && (!anorm.is_finite() || anorm <= 0.0)) {
                return Err(Error::new("joint origin and axis must be finite; quaternion and moving axis must be nonzero with finite norm"));
            }
        }
        let mut reached = vec![false; link_num];
        reached[0] = true;
        for joint in &model.joints {
            if joint.parent_link >= link_num || joint.child_link >= link_num {
                return Err(Error::new("joint link index is out of range"));
            }
            if !reached[joint.parent_link] || reached[joint.child_link] {
                return Err(Error::new("joints must form a tree in parent-before-child order rooted at link 0"));
            }
            reached[joint.child_link] = true;
            if joint.kind == JointKind::Prismatic && !allow_prismatic {
                return Err(Error::new("Rust backend currently supports fixed/revolute joints only; use the Python backend for prismatic or multi-DoF joints"));
            }
        }
        if reached.iter().any(|reached| !reached) {
            return Err(Error::new("all links must be connected to root link 0"));
        }
        let mut parent_link = vec![0usize; joint_num];
        let mut child_link = vec![0usize; joint_num];
        let mut q_index = vec![-1isize; joint_num];
        let mut is_prismatic = vec![false; joint_num];
        let mut axis = vec![[1.0, 0.0, 0.0]; joint_num];
        let mut origin_r = vec![eye3(); joint_num];
        let mut origin_p = vec![[0.0; 3]; joint_num];
        let mut link_ancestors = vec![Vec::new(); link_num];
        let mut link_child_joints = vec![Vec::new(); link_num];
        let mut dof = 0usize;
        for (i, joint) in model.joints.iter().enumerate() {
            parent_link[i] = joint.parent_link;
            child_link[i] = joint.child_link;
            link_child_joints[joint.parent_link].push(i);
            origin_p[i] = joint.position;
            origin_r[i] = quat_to_rot(joint.orientation);
            if joint.kind == JointKind::Fixed {
                link_ancestors[joint.child_link] = link_ancestors[joint.parent_link].clone();
                continue;
            }
            is_prismatic[i] = joint.kind == JointKind::Prismatic;
            axis[i] = normalize(joint.axis);
            q_index[i] = dof as isize;
            let mut ancestors = link_ancestors[joint.parent_link].clone();
            ancestors.push(dof);
            link_ancestors[joint.child_link] = ancestors;
            dof += 1;
        }
        let link_inertia = model.links.iter().map(spatial_inertia).collect();
        let link_motion_columns: Vec<Vec<usize>> = link_ancestors
            .iter()
            .map(|ancestors| {
                let mut cols = Vec::with_capacity(ancestors.len() * 3);
                for &qi in ancestors {
                    cols.push(3 * qi);
                    cols.push(3 * qi + 1);
                    cols.push(3 * qi + 2);
                }
                cols
            })
            .collect();
        let mut link_subtree_motion_columns = link_motion_columns.clone();
        for j in (0..joint_num).rev() {
            let parent = parent_link[j];
            let child = child_link[j];
            let child_cols = link_subtree_motion_columns[child].clone();
            merge_columns(&mut link_subtree_motion_columns[parent], &child_cols);
        }
        // Keep a topology cache for subtree reductions which must include
        // fixed links (the motion-column cache above intentionally cannot).
        let mut link_subtree_links: Vec<Vec<usize>> =
            (0..link_num).map(|link| vec![link]).collect();
        for j in (0..joint_num).rev() {
            let parent = parent_link[j];
            let child_links = link_subtree_links[child_link[j]].clone();
            link_subtree_links[parent].extend(child_links);
        }
        let mut topology_rank = vec![usize::MAX; link_num];
        topology_rank[0] = 0;
        for j in 0..joint_num {
            topology_rank[child_link[j]] = j + 1;
        }
        for links in &mut link_subtree_links {
            links.sort_unstable_by_key(|&link| topology_rank[link]);
        }

        let mut offset = 0;
        let joints = model.joints.iter().enumerate().map(|(id, joint)| {
            let dof = usize::from(joint.kind != JointKind::Fixed);
            let info = JointInfo { id, name: joint.name.clone(), kind: joint.kind,
                dof, dof_index: offset, parent_link: joint.parent_link, child_link: joint.child_link };
            offset += dof;
            info
        }).collect();
        let info = ModelInfo { dof, link_names: model.links.iter().map(|x| x.name.clone()).collect(),
            joints, supports_cmtm: !is_prismatic.iter().any(|x| *x) };
        Ok(RustCompiledRobot {
            info: std::sync::Arc::new(info),
            link_num,
            joint_num,
            dof,
            parent_link,
            child_link,
            q_index,
            is_prismatic,
            axis,
            origin_r,
            origin_p,
            link_inertia,
            link_ancestors,
            link_motion_columns,
            link_subtree_motion_columns,
            link_subtree_links,
            link_child_joints,
        })
    }
}
fn spatial_inertia(link: &LinkModel) -> [[f64; 6]; 6] {
    let LinkModel { mass, cog, inertia: iv, .. } = *link;
    let inertia = [
        [iv[0], iv[3], iv[4]],
        [iv[3], iv[1], iv[5]],
        [iv[4], iv[5], iv[2]],
    ];
    let c_hat = skew(cog);
    let c2 = mat3_mul(c_hat, c_hat);
    let mut out = [[0.0; 6]; 6];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = inertia[r][c] - mass * c2[r][c];
            out[r + 3][c + 3] = if r == c { mass } else { 0.0 };
            out[r + 3][c] = -mass * c_hat[r][c];
            out[r][c + 3] = mass * c_hat[r][c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::Workspace;

    fn model(kind: JointKind) -> RobotModel {
        RobotModel {
            links: vec![
                LinkModel { name: "world".into(), mass: 0.0, cog: [0.0; 3], inertia: [0.0; 6] },
                LinkModel { name: "body".into(), mass: 2.0, cog: [0.0; 3], inertia: [1.0, 2.0, 3.0, 0.0, 0.0, 0.0] },
            ],
            joints: vec![JointModel {
                name: "joint".into(),
                parent_link: 0, child_link: 1, kind, axis: [0.0, 0.0, 1.0],
                position: [0.0; 3], orientation: [1.0, 0.0, 0.0, 0.0],
            }],
        }
    }

    #[test]
    fn native_model_and_containers_without_python() {
        let robot = RustCompiledRobot::from_model(&model(JointKind::Revolute), false).unwrap();
        assert_eq!(robot.dof, 1);
        let mut ws = Workspace::new(&robot);
        robot.rnea_with_gravity_into(&[0.3], &[0.0], &[0.7], [0.0; 3], &mut ws);
        // Rotation about the center of mass: torque = Izz * angular acceleration.
        assert!((ws.tau[0] - 2.1).abs() < 1e-12);
        let mut outward = robot.create_outward_data(3).unwrap();
        robot.kinematics_cmtm_into(&[0.3, 0.0, 0.7], 3, &mut outward.dynamics.cmtm);
        assert!((outward.dynamics.cmtm.link_mat[16] - 0.3_f64.cos()).abs() < 1e-12);
        assert_eq!(robot.create_batch_outward_data(3, 2).unwrap().dynamics.len(), 2);
        assert!(!robot.create_selected_workspace(3).unwrap().ready);
        assert!(!robot.create_fast_data().has_kinematics);
        let mut aba = robot.create_aba_data();
        aba.prepare_into(&[0.3], &[0.0], [0.0; 3]).unwrap();
        aba.solve_into(&[2.1]).unwrap();
        assert!((aba.workspace.qdd[0] - 0.7).abs() < 1e-12);
    }

    #[test]
    fn native_prismatic_and_fixed_models() {
        let input = model(JointKind::Prismatic);
        assert!(RustCompiledRobot::from_model(&input, false).is_err());
        let robot = RustCompiledRobot::from_model(&input, true).unwrap();
        let mut ws = Workspace::new(&robot);
        robot.rnea_with_gravity_into(&[0.3], &[0.0], &[0.0], [0.0, 0.0, -9.81], &mut ws);
        assert!((ws.tau[0] - 19.62).abs() < 1e-12);
        let fixed = RustCompiledRobot::from_model(&model(JointKind::Fixed), false).unwrap();
        assert_eq!(fixed.dof, 0);
        assert_eq!(fixed.link_subtree_links[0], vec![0, 1]);
    }

    #[test]
    fn rejects_invalid_topology_before_indexing() {
        let mut input = model(JointKind::Revolute);
        input.joints[0].parent_link = 99;
        assert!(RustCompiledRobot::from_model(&input, false).is_err());
        input.joints[0].parent_link = 1;
        assert!(RustCompiledRobot::from_model(&input, false).is_err());
        input.joints.clear();
        assert!(RustCompiledRobot::from_model(&input, false).is_err());
        input.links.clear();
        assert!(RustCompiledRobot::from_model(&input, false).is_err());
    }

    #[test]
    fn branch_and_fixed_link_topology() {
        let mut input = model(JointKind::Revolute);
        input.links.extend([input.links[1].clone(), input.links[1].clone()]);
        input.links[2].name = "fixed_body".into();
        input.links[3].name = "branch_body".into();
        let mut fixed = input.joints[0].clone();
        fixed.name = "fixed_joint".into();
        fixed.parent_link = 1;
        fixed.child_link = 2;
        fixed.kind = JointKind::Fixed;
        let mut branch = input.joints[0].clone();
        branch.child_link = 3;
        branch.name = "branch_joint".into();
        input.joints.extend([fixed, branch]);
        let robot = RustCompiledRobot::from_model(&input, false).unwrap();
        assert_eq!(robot.dof, 2);
        assert_eq!(robot.link_ancestors, vec![vec![], vec![0], vec![0], vec![1]]);
        assert_eq!(robot.link_subtree_links[1], vec![1, 2]);
        assert_eq!(robot.link_subtree_links[0], vec![0, 1, 2, 3]);
    }
}
