use crate::error::{CoreResult, Error};

use crate::dynamics_outputs::{swap_spatial, DynamicsOutput, StateOutput};
use crate::spatial::*;
use crate::types::{
    RustAbaData, RustBatchOutwardData, RustCompiledRobot, RustOutwardData, RustSelectedWorkspace,
};
use crate::workspace::{CmtmWorkspace, DynamicsCmtmTangentWorkspace, DynamicsCmtmWorkspace};

impl RustOutwardData {
    /// World spatial motion: key_order 2 is velocity, 3 acceleration, etc.
    /// Includes derivatives of the moving frame; returns [angular, linear].
    pub fn world_link_vec(&self, link_id: usize, key_order: usize) -> CoreResult<Vec<f64>> {
        self.check_link_id(link_id)?;
        self.cmtm_vec_index(key_order)?;
        self.cmtm_source()?;
        Ok(world_spatial_value(
            &self.robot,
            &self.dynamics,
            self.order,
            link_id,
            false,
            key_order,
            self.has_cached_order1_dynamics,
        ))
    }
    /// World spatial motion: key_order 2 is velocity, 3 acceleration, etc.
    /// Includes derivatives of the moving frame; returns [angular, linear].
    pub fn world_joint_vec(&self, joint_id: usize, key_order: usize) -> CoreResult<Vec<f64>> {
        self.check_joint_id(joint_id)?;
        self.cmtm_vec_index(key_order)?;
        self.cmtm_source()?;
        Ok(world_spatial_value(
            &self.robot,
            &self.dynamics,
            self.order,
            joint_id,
            true,
            key_order,
            self.has_cached_order1_dynamics,
        ))
    }

    pub(crate) fn check_link_id(&self, link_id: usize) -> CoreResult<()> {
        if link_id >= self.robot.link_num {
            return Err(Error::new(format!(
                "invalid link_id: {link_id}. Must be < {}",
                self.robot.link_num
            )));
        }
        Ok(())
    }

    pub(crate) fn check_joint_id(&self, joint_id: usize) -> CoreResult<()> {
        if joint_id >= self.robot.joint_num {
            return Err(Error::new(format!(
                "invalid joint_id: {joint_id}. Must be < {}",
                self.robot.joint_num
            )));
        }
        Ok(())
    }

    pub(crate) fn check_dynamics_computed(&self) -> CoreResult<()> {
        if !self.has_dynamics {
            return Err(Error::new(
                "compute_dynamics must be called before reading dynamics values",
            ));
        }
        Ok(())
    }

    pub(crate) fn cmtm_source(&self) -> CoreResult<&CmtmWorkspace> {
        if self.has_kinematics {
            return Ok(&self.dynamics.cmtm);
        }
        Err(Error::new(
            "compute_kinematics or compute_dynamics must be called before reading kinematics values",
        ))
    }

    pub(crate) fn cmtm_vec_index(&self, key_order: usize) -> CoreResult<usize> {
        if key_order < 2 || key_order > self.order {
            return Err(Error::new(format!(
                "invalid kinematics key_order: {key_order}. Must be in 2..={}",
                self.order
            )));
        }
        Ok(key_order - 2)
    }

    pub(crate) fn momentum_vec_index(&self, key_order: usize) -> CoreResult<usize> {
        if key_order < 1 || key_order > self.dynamics_order + 1 {
            return Err(Error::new(format!(
                "invalid momentum key_order: {key_order}. Must be in 1..={}",
                self.dynamics_order + 1
            )));
        }
        Ok(key_order - 1)
    }

    pub(crate) fn force_vec_index(&self, key_order: usize) -> CoreResult<usize> {
        if key_order < 1 || key_order > self.dynamics_order {
            return Err(Error::new(format!(
                "invalid force key_order: {key_order}. Must be in 1..={}",
                self.dynamics_order
            )));
        }
        Ok(key_order - 1)
    }
}

impl RustBatchOutwardData {
    /// World spatial motion: key_order 2 is velocity, 3 acceleration, etc.
    /// Includes derivatives of the moving frame; returns [angular, linear].
    pub fn world_link_vec(
        &self,
        sample: usize,
        link_id: usize,
        key_order: usize,
    ) -> CoreResult<Vec<f64>> {
        if sample >= self.batch {
            return Err(Error::new("sample index is out of range"));
        }
        self.check_link_id(link_id)?;
        self.cmtm_vec_index(key_order)?;
        self.cmtm_source(sample)?;
        Ok(world_spatial_value(
            &self.robot,
            &self.dynamics[sample],
            self.order,
            link_id,
            false,
            key_order,
            self.has_cached_order1_dynamics,
        ))
    }
    /// World spatial motion: key_order 2 is velocity, 3 acceleration, etc.
    /// Includes derivatives of the moving frame; returns [angular, linear].
    pub fn world_joint_vec(
        &self,
        sample: usize,
        joint_id: usize,
        key_order: usize,
    ) -> CoreResult<Vec<f64>> {
        if sample >= self.batch {
            return Err(Error::new("sample index is out of range"));
        }
        self.check_joint_id(joint_id)?;
        self.cmtm_vec_index(key_order)?;
        self.cmtm_source(sample)?;
        Ok(world_spatial_value(
            &self.robot,
            &self.dynamics[sample],
            self.order,
            joint_id,
            true,
            key_order,
            self.has_cached_order1_dynamics,
        ))
    }

    pub(crate) fn check_motion_shape(&self, shape: &[usize]) -> CoreResult<()> {
        let expected = self.robot.dof * self.order;
        if shape != [self.batch, expected] {
            return Err(Error::new(format!(
                "motions must have shape ({}, {}), got {:?}",
                self.batch, expected, shape
            )));
        }
        Ok(())
    }

    pub(crate) fn check_link_id(&self, link_id: usize) -> CoreResult<()> {
        if link_id >= self.robot.link_num {
            return Err(Error::new(format!(
                "invalid link_id: {link_id}. Must be < {}",
                self.robot.link_num
            )));
        }
        Ok(())
    }

    pub(crate) fn check_joint_id(&self, joint_id: usize) -> CoreResult<()> {
        if joint_id >= self.robot.joint_num {
            return Err(Error::new(format!(
                "invalid joint_id: {joint_id}. Must be < {}",
                self.robot.joint_num
            )));
        }
        Ok(())
    }

    pub(crate) fn check_dynamics_computed(&self) -> CoreResult<()> {
        if !self.has_dynamics {
            return Err(Error::new(
                "compute_dynamics must be called before reading dynamics values",
            ));
        }
        Ok(())
    }

    pub(crate) fn cmtm_source(&self, sample: usize) -> CoreResult<&CmtmWorkspace> {
        if self.has_kinematics {
            return Ok(&self.dynamics[sample].cmtm);
        }
        Err(Error::new(
            "compute_kinematics or compute_dynamics must be called before reading kinematics values",
        ))
    }

    pub(crate) fn cmtm_vec_index(&self, key_order: usize) -> CoreResult<usize> {
        if key_order < 2 || key_order > self.order {
            return Err(Error::new(format!(
                "invalid kinematics key_order: {key_order}. Must be in 2..={}",
                self.order
            )));
        }
        Ok(key_order - 2)
    }

    pub(crate) fn momentum_vec_index(&self, key_order: usize) -> CoreResult<usize> {
        if key_order < 1 || key_order > self.dynamics_order + 1 {
            return Err(Error::new(format!(
                "invalid momentum key_order: {key_order}. Must be in 1..={}",
                self.dynamics_order + 1
            )));
        }
        Ok(key_order - 1)
    }

    pub(crate) fn force_vec_index(&self, key_order: usize) -> CoreResult<usize> {
        if key_order < 1 || key_order > self.dynamics_order {
            return Err(Error::new(format!(
                "invalid force key_order: {key_order}. Must be in 1..={}",
                self.dynamics_order
            )));
        }
        Ok(key_order - 1)
    }
}

impl RustAbaData {
    pub fn prepare_into(
        &mut self,
        q: &[f64],
        v: &[f64],
        gravity: [f64; 3],
    ) -> crate::error::CoreResult<()> {
        if q.len() != self.robot.dof || v.len() != self.robot.dof {
            return Err(Error::new("q/v length must match robot dof"));
        }
        check_gravity(gravity)?;
        let changed = !self.prepared
            || self.bias_q.as_slice() != q
            || self.bias_v.as_slice() != v
            || self.bias_gravity != gravity;
        if changed {
            let zero = vec![0.0; self.robot.dof];
            self.robot
                .aba_with_gravity_into(q, v, &zero, gravity, &mut self.workspace)
                .map_err(Error::new)?;
            self.workspace.bias_qdd.copy_from_slice(&self.workspace.qdd);
            self.robot
                .aba_factorize_mass_into(q, &mut self.workspace)
                .map_err(Error::new)?;
            self.factor_q.clear();
            self.factor_q.extend_from_slice(q);
            self.bias_q.clear();
            self.bias_q.extend_from_slice(q);
            self.bias_v.clear();
            self.bias_v.extend_from_slice(v);
            self.bias_gravity = gravity;
            self.prepared = true;
        }
        Ok(())
    }

    pub fn solve_into(&mut self, tau: &[f64]) -> crate::error::CoreResult<()> {
        if !self.prepared {
            return Err(Error::new("call prepare before solve"));
        }
        self.robot
            .aba_solve_mass_into(tau, &mut self.workspace)
            .map_err(Error::new)?;
        for i in 0..self.robot.dof {
            self.workspace.qdd[i] += self.workspace.bias_qdd[i];
        }
        Ok(())
    }
}

pub(crate) fn cmtm_world_wrench_value(
    elem_mat: [[f64; 4]; 4],
    elem_vecs: &[f64],
    raw_vecs: &[f64],
    key_order: usize,
) -> Vec<f64> {
    let mut fact = vec![1.0; key_order.max(1)];
    fill_factorial_table(&mut fact);
    let mut cm_rhs = vec![0.0; key_order * 6];
    for k in 0..key_order {
        let scale = 1.0 / fact[k];
        for i in 0..6 {
            cm_rhs[k * 6 + i] = raw_vecs[k * 6 + i] * scale;
        }
    }
    let mut blocks = vec![[[0.0; 6]; 6]; key_order];
    let mut out_cm = vec![0.0; key_order * 6];
    cmtm_apply_mat_adj_wrench_with_blocks_into(
        elem_mat,
        elem_vecs,
        &cm_rhs,
        key_order,
        &fact,
        &mut blocks,
        &mut out_cm,
    );
    let start = (key_order - 1) * 6;
    let scale = fact[key_order - 1];
    out_cm[start..start + 6]
        .iter()
        .map(|value| value * scale)
        .collect()
}
// Ad_motion(T) = P Ad_wrench(T) P. Reuse the same ordinary-derivative
// series transform as the selected world-motion tangent/reverse kernels.
fn world_spatial_value(
    robot: &RustCompiledRobot,
    ws: &DynamicsCmtmWorkspace,
    order: usize,
    owner: usize,
    joint: bool,
    key_order: usize,
    cached_order1: bool,
) -> Vec<f64> {
    let n = key_order - 1;
    let link = if joint {
        robot.child_link[owner]
    } else {
        owner
    };
    let mat = mat4_from_flat(&ws.cmtm.link_mat, link);
    let frame_vecs = cmtm_vecs_slice(&ws.cmtm.link_vecs, link, order);
    let cached;
    let raw = if joint && cached_order1 {
        cached = (0..n)
            .flat_map(|i| order1_joint_vec_value(robot, ws, owner, i))
            .collect::<Vec<_>>();
        cached.as_slice()
    } else {
        let series = if joint {
            &ws.cmtm.joint_vecs
        } else {
            &ws.cmtm.link_vecs
        };
        &cmtm_vecs_slice(series, owner, order)[..n * 6]
    };
    let swapped = swap_spatial(raw);
    swap_spatial(&cmtm_world_wrench_value(mat, frame_vecs, &swapped, n))
}

pub(crate) fn order1_link_momentum_value(
    robot: &RustCompiledRobot,
    ws: &DynamicsCmtmWorkspace,
    link_id: usize,
    vec_index: usize,
) -> [f64; 6] {
    let link_vel = cmtm_vecs_slice(&ws.cmtm.link_vecs, link_id, 3);
    mat6_vec6(
        robot.link_inertia[link_id],
        vec6_from_flat(link_vel, vec_index),
    )
}
pub(crate) fn order1_link_force_value(
    robot: &RustCompiledRobot,
    ws: &DynamicsCmtmWorkspace,
    link_id: usize,
) -> [f64; 6] {
    let link_vel = cmtm_vecs_slice(&ws.cmtm.link_vecs, link_id, 3);
    let v0 = vec6_from_flat(link_vel, 0);
    let m0 = mat6_vec6(robot.link_inertia[link_id], v0);
    let m1 = mat6_vec6(robot.link_inertia[link_id], vec6_from_flat(link_vel, 1));
    add6(m1, hat_adj_wrench_vec6(v0, m0))
}
pub(crate) fn order1_joint_force_value(
    robot: &RustCompiledRobot,
    ws: &DynamicsCmtmWorkspace,
    joint_id: usize,
) -> [f64; 6] {
    let child = robot.child_link[joint_id];
    let link_vel = cmtm_vecs_slice(&ws.cmtm.link_vecs, child, 3);
    let v0 = vec6_from_flat(link_vel, 0);
    let start = joint_id * 2 * 6;
    let m0 = vec6_from_flat(&ws.joint_momentum[start..start + 12], 0);
    let m1 = vec6_from_flat(&ws.joint_momentum[start..start + 12], 1);
    add6(m1, hat_adj_wrench_vec6(v0, m0))
}
pub(crate) fn order1_joint_mat_value(
    robot: &RustCompiledRobot,
    ws: &DynamicsCmtmWorkspace,
    joint_id: usize,
) -> [[f64; 4]; 4] {
    let q_index = robot.q_index[joint_id];
    if q_index < 0 {
        return eye4();
    }
    let q = ws.cached_motion[q_index as usize * 3];
    mat4_from_rot_pos(rot_axis(robot.axis[joint_id], q), [0.0; 3])
}
pub(crate) fn order1_joint_vec_value(
    robot: &RustCompiledRobot,
    ws: &DynamicsCmtmWorkspace,
    joint_id: usize,
    vec_index: usize,
) -> [f64; 6] {
    let q_index = robot.q_index[joint_id];
    if q_index < 0 {
        return [0.0; 6];
    }
    let value = ws.cached_motion[q_index as usize * 3 + vec_index + 1];
    [
        robot.axis[joint_id][0] * value,
        robot.axis[joint_id][1] * value,
        robot.axis[joint_id][2] * value,
        0.0,
        0.0,
        0.0,
    ]
}
impl RustOutwardData {
    pub fn compute_kinematics(&mut self, motion: &[f64]) -> CoreResult<()> {
        self.robot.check_cmtm_motion(motion, self.order)?;
        self.robot
            .kinematics_cmtm_into(motion, self.order, &mut self.dynamics.cmtm);
        self.has_kinematics = true;
        self.has_dynamics = false;
        self.has_full_dynamics = false;
        self.has_cached_order1_dynamics = false;
        Ok(())
    }
    pub fn compute_dynamics(&mut self, motion: &[f64], gravity: [f64; 3]) -> CoreResult<()> {
        if self.order < 2 {
            return Err(Error::new("dynamics data requires order >= 2"));
        }
        check_gravity(gravity)?;
        self.robot.check_cmtm_motion(motion, self.order)?;
        self.dynamics
            .ensure_dynamics(&self.robot, self.dynamics_order);
        if self.order == 3 && self.dynamics_order == 1 && gravity == [0.0; 3] {
            self.robot
                .dynamics_cmtm_order1_cached_into(motion, &mut self.dynamics);
            self.has_cached_order1_dynamics = true;
        } else {
            self.robot
                .dynamics_cmtm_into(motion, self.dynamics_order, gravity, &mut self.dynamics);
            self.has_cached_order1_dynamics = false;
        }
        self.has_kinematics = true;
        self.has_dynamics = true;
        self.has_full_dynamics = true;
        Ok(())
    }
    pub fn compute_dynamics_minimal(
        &mut self,
        motion: &[f64],
        gravity: [f64; 3],
    ) -> CoreResult<()> {
        if self.order < 2 {
            return Err(Error::new("dynamics data requires order >= 2"));
        }
        check_gravity(gravity)?;
        self.robot.check_cmtm_motion(motion, self.order)?;
        self.dynamics
            .ensure_dynamics(&self.robot, self.dynamics_order);
        self.robot.dynamics_cmtm_minimal_into(
            motion,
            self.dynamics_order,
            gravity,
            &mut self.dynamics,
        );
        self.has_kinematics = true;
        self.has_dynamics = true;
        self.has_full_dynamics = false;
        self.has_cached_order1_dynamics = false;
        Ok(())
    }
}
impl RustBatchOutwardData {
    pub fn compute_kinematics(&mut self, motions: &[f64]) -> CoreResult<()> {
        if motions.len() != self.batch * self.robot.dof * self.order {
            return Err(Error::new(
                "motions length must match batch * robot dof * order",
            ));
        }
        let motion_len = self.robot.dof * self.order;
        for sample in 0..self.batch {
            let start = sample * motion_len;
            let end = start + motion_len;
            self.robot.kinematics_cmtm_into(
                &motions[start..end],
                self.order,
                &mut self.dynamics[sample].cmtm,
            );
        }
        self.has_kinematics = true;
        self.has_dynamics = false;
        self.has_full_dynamics = false;
        self.has_cached_order1_dynamics = false;
        Ok(())
    }
    pub fn compute_dynamics(&mut self, motions: &[f64], gravity: [f64; 3]) -> CoreResult<()> {
        if self.order < 2 {
            return Err(Error::new("dynamics data requires order >= 2"));
        }
        check_gravity(gravity)?;
        if motions.len() != self.batch * self.robot.dof * self.order {
            return Err(Error::new(
                "motions length must match batch * robot dof * order",
            ));
        }
        let motion_len = self.robot.dof * self.order;
        for sample in 0..self.batch {
            let start = sample * motion_len;
            let end = start + motion_len;
            self.dynamics[sample].ensure_dynamics(&self.robot, self.dynamics_order);
            if self.order == 3 && self.dynamics_order == 1 && gravity == [0.0; 3] {
                self.robot.dynamics_cmtm_order1_cached_into(
                    &motions[start..end],
                    &mut self.dynamics[sample],
                );
            } else {
                self.robot.dynamics_cmtm_into(
                    &motions[start..end],
                    self.dynamics_order,
                    gravity,
                    &mut self.dynamics[sample],
                );
            }
        }
        self.has_kinematics = true;
        self.has_dynamics = true;
        self.has_full_dynamics = true;
        self.has_cached_order1_dynamics =
            self.order == 3 && self.dynamics_order == 1 && gravity == [0.0; 3];
        Ok(())
    }
    pub fn compute_dynamics_minimal(
        &mut self,
        motions: &[f64],
        gravity: [f64; 3],
    ) -> CoreResult<()> {
        if self.order < 2 {
            return Err(Error::new("dynamics data requires order >= 2"));
        }
        check_gravity(gravity)?;
        if motions.len() != self.batch * self.robot.dof * self.order {
            return Err(Error::new(
                "motions length must match batch * robot dof * order",
            ));
        }
        let motion_len = self.robot.dof * self.order;
        for sample in 0..self.batch {
            let start = sample * motion_len;
            let end = start + motion_len;
            self.dynamics[sample].ensure_dynamics(&self.robot, self.dynamics_order);
            self.robot.dynamics_cmtm_minimal_into(
                &motions[start..end],
                self.dynamics_order,
                gravity,
                &mut self.dynamics[sample],
            );
        }
        self.has_kinematics = true;
        self.has_dynamics = true;
        self.has_full_dynamics = false;
        self.has_cached_order1_dynamics = false;
        Ok(())
    }
}
impl RustOutwardData {
    pub fn link_mat(&self, link_id: usize) -> CoreResult<Vec<f64>> {
        self.check_link_id(link_id)?;
        let ws = self.cmtm_source()?;
        let start = link_id * 16;
        Ok(ws.link_mat[start..start + 16].to_vec())
    }
    pub fn joint_mat(&self, joint_id: usize) -> CoreResult<Vec<f64>> {
        self.check_joint_id(joint_id)?;
        if self.has_cached_order1_dynamics {
            let mat = order1_joint_mat_value(&self.robot, &self.dynamics, joint_id);
            let mut out = Vec::with_capacity(16);
            for row in mat {
                out.extend_from_slice(&row);
            }
            return Ok(out);
        }
        let ws = self.cmtm_source()?;
        let start = joint_id * 16;
        Ok(ws.joint_mat[start..start + 16].to_vec())
    }
    pub fn link_vec(&self, link_id: usize, key_order: usize) -> CoreResult<Vec<f64>> {
        self.check_link_id(link_id)?;
        let vec_index = self.cmtm_vec_index(key_order)?;
        let ws = self.cmtm_source()?;
        let start = (link_id * (self.order - 1) + vec_index) * 6;
        Ok(ws.link_vecs[start..start + 6].to_vec())
    }
    pub fn joint_vec(&self, joint_id: usize, key_order: usize) -> CoreResult<Vec<f64>> {
        self.check_joint_id(joint_id)?;
        let vec_index = self.cmtm_vec_index(key_order)?;
        if self.has_cached_order1_dynamics {
            return Ok(
                order1_joint_vec_value(&self.robot, &self.dynamics, joint_id, vec_index).to_vec(),
            );
        }
        let ws = self.cmtm_source()?;
        let start = (joint_id * (self.order - 1) + vec_index) * 6;
        Ok(ws.joint_vecs[start..start + 6].to_vec())
    }
    pub fn link_momentum(&self, link_id: usize, key_order: usize) -> CoreResult<Vec<f64>> {
        self.check_link_id(link_id)?;
        self.check_dynamics_computed()?;
        if !self.has_full_dynamics {
            return Err(Error::new("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values"));
        }
        let vec_index = self.momentum_vec_index(key_order)?;
        if self.has_cached_order1_dynamics {
            return Ok(
                order1_link_momentum_value(&self.robot, &self.dynamics, link_id, vec_index)
                    .to_vec(),
            );
        }
        let start = (link_id * (self.dynamics_order + 1) + vec_index) * 6;
        Ok(self.dynamics.link_momentum[start..start + 6].to_vec())
    }
    pub fn joint_momentum(&self, joint_id: usize, key_order: usize) -> CoreResult<Vec<f64>> {
        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        if !self.has_full_dynamics {
            return Err(Error::new("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values"));
        }
        let vec_index = self.momentum_vec_index(key_order)?;
        let start = (joint_id * (self.dynamics_order + 1) + vec_index) * 6;
        Ok(self.dynamics.joint_momentum[start..start + 6].to_vec())
    }
    pub fn world_link_momentum(&self, link_id: usize, key_order: usize) -> CoreResult<Vec<f64>> {
        self.check_link_id(link_id)?;
        self.check_dynamics_computed()?;
        if !self.has_full_dynamics {
            return Err(Error::new("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values"));
        }
        self.momentum_vec_index(key_order)?;
        let ws = self.cmtm_source()?;
        let mat = mat4_from_flat(&ws.link_mat, link_id);
        let vecs = cmtm_vecs_slice(&ws.link_vecs, link_id, self.order);
        if self.has_cached_order1_dynamics {
            let mut raw = [0.0; 12];
            for index in 0..key_order {
                let value = order1_link_momentum_value(&self.robot, &self.dynamics, link_id, index);
                raw[index * 6..index * 6 + 6].copy_from_slice(&value);
            }
            let out = cmtm_world_wrench_value(mat, vecs, &raw[..key_order * 6], key_order);
            return Ok(out);
        }
        let start = link_id * (self.dynamics_order + 1) * 6;
        let out = cmtm_world_wrench_value(
            mat,
            vecs,
            &self.dynamics.link_momentum[start..start + key_order * 6],
            key_order,
        );
        Ok(out)
    }
    pub fn world_joint_momentum(&self, joint_id: usize, key_order: usize) -> CoreResult<Vec<f64>> {
        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        if !self.has_full_dynamics {
            return Err(Error::new("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values"));
        }
        self.momentum_vec_index(key_order)?;
        let link_id = self.robot.child_link[joint_id];
        let ws = self.cmtm_source()?;
        let mat = mat4_from_flat(&ws.link_mat, link_id);
        let vecs = cmtm_vecs_slice(&ws.link_vecs, link_id, self.order);
        let start = joint_id * (self.dynamics_order + 1) * 6;
        let out = cmtm_world_wrench_value(
            mat,
            vecs,
            &self.dynamics.joint_momentum[start..start + key_order * 6],
            key_order,
        );
        Ok(out)
    }
    pub fn link_force(&self, link_id: usize, key_order: usize) -> CoreResult<Vec<f64>> {
        self.check_link_id(link_id)?;
        self.check_dynamics_computed()?;
        if !self.has_full_dynamics {
            return Err(Error::new("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values"));
        }
        let vec_index = self.force_vec_index(key_order)?;
        if self.has_cached_order1_dynamics {
            return Ok(order1_link_force_value(&self.robot, &self.dynamics, link_id).to_vec());
        }
        let start = (link_id * self.dynamics_order + vec_index) * 6;
        Ok(self.dynamics.link_force[start..start + 6].to_vec())
    }
    pub fn world_link_force(&self, link_id: usize, key_order: usize) -> CoreResult<Vec<f64>> {
        self.check_link_id(link_id)?;
        self.check_dynamics_computed()?;
        if !self.has_full_dynamics {
            return Err(Error::new("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values"));
        }
        self.force_vec_index(key_order)?;
        let ws = self.cmtm_source()?;
        let mat = mat4_from_flat(&ws.link_mat, link_id);
        let vecs = cmtm_vecs_slice(&ws.link_vecs, link_id, self.order);
        if self.has_cached_order1_dynamics {
            let force = order1_link_force_value(&self.robot, &self.dynamics, link_id);
            let out = cmtm_world_wrench_value(mat, vecs, &force, key_order);
            return Ok(out);
        }
        let start = link_id * self.dynamics_order * 6;
        let out = cmtm_world_wrench_value(
            mat,
            vecs,
            &self.dynamics.link_force[start..start + key_order * 6],
            key_order,
        );
        Ok(out)
    }
    pub fn world_joint_force(&self, joint_id: usize, key_order: usize) -> CoreResult<Vec<f64>> {
        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        if !self.has_full_dynamics {
            return Err(Error::new("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values"));
        }
        self.force_vec_index(key_order)?;
        let link_id = self.robot.child_link[joint_id];
        let ws = self.cmtm_source()?;
        let mat = mat4_from_flat(&ws.link_mat, link_id);
        let vecs = cmtm_vecs_slice(&ws.link_vecs, link_id, self.order);
        if self.has_cached_order1_dynamics {
            let force = order1_joint_force_value(&self.robot, &self.dynamics, joint_id);
            let out = cmtm_world_wrench_value(mat, vecs, &force, key_order);
            return Ok(out);
        }
        let start = joint_id * self.dynamics_order * 6;
        let out = cmtm_world_wrench_value(
            mat,
            vecs,
            &self.dynamics.joint_force[start..start + key_order * 6],
            key_order,
        );
        Ok(out)
    }
    pub fn joint_force(&self, joint_id: usize, key_order: usize) -> CoreResult<Vec<f64>> {
        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        if !self.has_full_dynamics {
            return Err(Error::new("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values"));
        }
        let vec_index = self.force_vec_index(key_order)?;
        if self.has_cached_order1_dynamics {
            return Ok(order1_joint_force_value(&self.robot, &self.dynamics, joint_id).to_vec());
        }
        let start = (joint_id * self.dynamics_order + vec_index) * 6;
        Ok(self.dynamics.joint_force[start..start + 6].to_vec())
    }
    pub fn joint_torque(&self, joint_id: usize, key_order: usize) -> CoreResult<Vec<f64>> {
        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        let vec_index = self.force_vec_index(key_order)?;
        let start = joint_id * self.dynamics_order + vec_index;
        Ok(vec![self.dynamics.joint_torque[start]])
    }
}
impl RustBatchOutwardData {
    pub fn link_mat(&self, sample: usize, link_id: usize) -> CoreResult<Vec<f64>> {
        if sample >= self.batch {
            return Err(Error::new("sample index is out of range"));
        }

        self.check_link_id(link_id)?;
        let ws = self.cmtm_source(sample)?;
        let start = link_id * 16;
        Ok(ws.link_mat[start..start + 16].to_vec())
    }
    pub fn joint_mat(&self, sample: usize, joint_id: usize) -> CoreResult<Vec<f64>> {
        if sample >= self.batch {
            return Err(Error::new("sample index is out of range"));
        }

        self.check_joint_id(joint_id)?;
        if self.has_cached_order1_dynamics {
            let mat = order1_joint_mat_value(&self.robot, &self.dynamics[sample], joint_id);
            let mut out = Vec::with_capacity(16);
            for row in mat {
                out.extend_from_slice(&row);
            }
            return Ok(out);
        }
        let ws = self.cmtm_source(sample)?;
        let start = joint_id * 16;
        Ok(ws.joint_mat[start..start + 16].to_vec())
    }
    pub fn link_vec(
        &self,
        sample: usize,
        link_id: usize,
        key_order: usize,
    ) -> CoreResult<Vec<f64>> {
        if sample >= self.batch {
            return Err(Error::new("sample index is out of range"));
        }

        self.check_link_id(link_id)?;
        let vec_index = self.cmtm_vec_index(key_order)?;
        let ws = self.cmtm_source(sample)?;
        let start = (link_id * (self.order - 1) + vec_index) * 6;
        Ok(ws.link_vecs[start..start + 6].to_vec())
    }
    pub fn joint_vec(
        &self,
        sample: usize,
        joint_id: usize,
        key_order: usize,
    ) -> CoreResult<Vec<f64>> {
        if sample >= self.batch {
            return Err(Error::new("sample index is out of range"));
        }

        self.check_joint_id(joint_id)?;
        let vec_index = self.cmtm_vec_index(key_order)?;
        if self.has_cached_order1_dynamics {
            return Ok(order1_joint_vec_value(
                &self.robot,
                &self.dynamics[sample],
                joint_id,
                vec_index,
            )
            .to_vec());
        }
        let ws = self.cmtm_source(sample)?;
        let start = (joint_id * (self.order - 1) + vec_index) * 6;
        Ok(ws.joint_vecs[start..start + 6].to_vec())
    }
    pub fn link_momentum(
        &self,
        sample: usize,
        link_id: usize,
        key_order: usize,
    ) -> CoreResult<Vec<f64>> {
        if sample >= self.batch {
            return Err(Error::new("sample index is out of range"));
        }

        self.check_link_id(link_id)?;
        self.check_dynamics_computed()?;
        if !self.has_full_dynamics {
            return Err(Error::new("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values"));
        }
        let vec_index = self.momentum_vec_index(key_order)?;
        if self.has_cached_order1_dynamics {
            return Ok(order1_link_momentum_value(
                &self.robot,
                &self.dynamics[sample],
                link_id,
                vec_index,
            )
            .to_vec());
        }
        let start = (link_id * (self.dynamics_order + 1) + vec_index) * 6;
        Ok(self.dynamics[sample].link_momentum[start..start + 6].to_vec())
    }
    pub fn joint_momentum(
        &self,
        sample: usize,
        joint_id: usize,
        key_order: usize,
    ) -> CoreResult<Vec<f64>> {
        if sample >= self.batch {
            return Err(Error::new("sample index is out of range"));
        }

        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        if !self.has_full_dynamics {
            return Err(Error::new("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values"));
        }
        let vec_index = self.momentum_vec_index(key_order)?;
        let start = (joint_id * (self.dynamics_order + 1) + vec_index) * 6;
        Ok(self.dynamics[sample].joint_momentum[start..start + 6].to_vec())
    }
    pub fn world_link_momentum(
        &self,
        sample: usize,
        link_id: usize,
        key_order: usize,
    ) -> CoreResult<Vec<f64>> {
        if sample >= self.batch {
            return Err(Error::new("sample index is out of range"));
        }

        self.check_link_id(link_id)?;
        self.check_dynamics_computed()?;
        if !self.has_full_dynamics {
            return Err(Error::new("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values"));
        }
        self.momentum_vec_index(key_order)?;
        let ws = self.cmtm_source(sample)?;
        let mat = mat4_from_flat(&ws.link_mat, link_id);
        let vecs = cmtm_vecs_slice(&ws.link_vecs, link_id, self.order);
        if self.has_cached_order1_dynamics {
            let mut raw = [0.0; 12];
            for index in 0..key_order {
                let value =
                    order1_link_momentum_value(&self.robot, &self.dynamics[sample], link_id, index);
                raw[index * 6..index * 6 + 6].copy_from_slice(&value);
            }
            let out = cmtm_world_wrench_value(mat, vecs, &raw[..key_order * 6], key_order);
            return Ok(out);
        }
        let start = link_id * (self.dynamics_order + 1) * 6;
        let out = cmtm_world_wrench_value(
            mat,
            vecs,
            &self.dynamics[sample].link_momentum[start..start + key_order * 6],
            key_order,
        );
        Ok(out)
    }
    pub fn world_joint_momentum(
        &self,
        sample: usize,
        joint_id: usize,
        key_order: usize,
    ) -> CoreResult<Vec<f64>> {
        if sample >= self.batch {
            return Err(Error::new("sample index is out of range"));
        }

        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        if !self.has_full_dynamics {
            return Err(Error::new("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values"));
        }
        self.momentum_vec_index(key_order)?;
        let link_id = self.robot.child_link[joint_id];
        let ws = self.cmtm_source(sample)?;
        let mat = mat4_from_flat(&ws.link_mat, link_id);
        let vecs = cmtm_vecs_slice(&ws.link_vecs, link_id, self.order);
        let start = joint_id * (self.dynamics_order + 1) * 6;
        let out = cmtm_world_wrench_value(
            mat,
            vecs,
            &self.dynamics[sample].joint_momentum[start..start + key_order * 6],
            key_order,
        );
        Ok(out)
    }
    pub fn link_force(
        &self,
        sample: usize,
        link_id: usize,
        key_order: usize,
    ) -> CoreResult<Vec<f64>> {
        if sample >= self.batch {
            return Err(Error::new("sample index is out of range"));
        }

        self.check_link_id(link_id)?;
        self.check_dynamics_computed()?;
        if !self.has_full_dynamics {
            return Err(Error::new("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values"));
        }
        let vec_index = self.force_vec_index(key_order)?;
        if self.has_cached_order1_dynamics {
            return Ok(
                order1_link_force_value(&self.robot, &self.dynamics[sample], link_id).to_vec(),
            );
        }
        let start = (link_id * self.dynamics_order + vec_index) * 6;
        Ok(self.dynamics[sample].link_force[start..start + 6].to_vec())
    }
    pub fn world_link_force(
        &self,
        sample: usize,
        link_id: usize,
        key_order: usize,
    ) -> CoreResult<Vec<f64>> {
        if sample >= self.batch {
            return Err(Error::new("sample index is out of range"));
        }

        self.check_link_id(link_id)?;
        self.check_dynamics_computed()?;
        if !self.has_full_dynamics {
            return Err(Error::new("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values"));
        }
        self.force_vec_index(key_order)?;
        let ws = self.cmtm_source(sample)?;
        let mat = mat4_from_flat(&ws.link_mat, link_id);
        let vecs = cmtm_vecs_slice(&ws.link_vecs, link_id, self.order);
        if self.has_cached_order1_dynamics {
            let force = order1_link_force_value(&self.robot, &self.dynamics[sample], link_id);
            let out = cmtm_world_wrench_value(mat, vecs, &force, key_order);
            return Ok(out);
        }
        let start = link_id * self.dynamics_order * 6;
        let out = cmtm_world_wrench_value(
            mat,
            vecs,
            &self.dynamics[sample].link_force[start..start + key_order * 6],
            key_order,
        );
        Ok(out)
    }
    pub fn world_joint_force(
        &self,
        sample: usize,
        joint_id: usize,
        key_order: usize,
    ) -> CoreResult<Vec<f64>> {
        if sample >= self.batch {
            return Err(Error::new("sample index is out of range"));
        }

        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        if !self.has_full_dynamics {
            return Err(Error::new("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values"));
        }
        self.force_vec_index(key_order)?;
        let link_id = self.robot.child_link[joint_id];
        let ws = self.cmtm_source(sample)?;
        let mat = mat4_from_flat(&ws.link_mat, link_id);
        let vecs = cmtm_vecs_slice(&ws.link_vecs, link_id, self.order);
        if self.has_cached_order1_dynamics {
            let force = order1_joint_force_value(&self.robot, &self.dynamics[sample], joint_id);
            let out = cmtm_world_wrench_value(mat, vecs, &force, key_order);
            return Ok(out);
        }
        let start = joint_id * self.dynamics_order * 6;
        let out = cmtm_world_wrench_value(
            mat,
            vecs,
            &self.dynamics[sample].joint_force[start..start + key_order * 6],
            key_order,
        );
        Ok(out)
    }
    pub fn joint_force(
        &self,
        sample: usize,
        joint_id: usize,
        key_order: usize,
    ) -> CoreResult<Vec<f64>> {
        if sample >= self.batch {
            return Err(Error::new("sample index is out of range"));
        }

        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        if !self.has_full_dynamics {
            return Err(Error::new("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values"));
        }
        let vec_index = self.force_vec_index(key_order)?;
        if self.has_cached_order1_dynamics {
            return Ok(
                order1_joint_force_value(&self.robot, &self.dynamics[sample], joint_id).to_vec(),
            );
        }
        let start = (joint_id * self.dynamics_order + vec_index) * 6;
        Ok(self.dynamics[sample].joint_force[start..start + 6].to_vec())
    }
    pub fn joint_torque(
        &self,
        sample: usize,
        joint_id: usize,
        key_order: usize,
    ) -> CoreResult<Vec<f64>> {
        if sample >= self.batch {
            return Err(Error::new("sample index is out of range"));
        }

        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        let vec_index = self.force_vec_index(key_order)?;
        let start = joint_id * self.dynamics_order + vec_index;
        Ok(vec![self.dynamics[sample].joint_torque[start]])
    }
}
impl RustSelectedWorkspace {
    fn prepare_selected_primal(&mut self, motion: &[f64], outputs: &[DynamicsOutput],
                               batch: usize, gravity: [f64; 3]) -> CoreResult<()> {
        let input_len = self.robot.dof * self.order;
        if motion.len() != checked_size(&[batch, input_len])? {
            return Err(Error::new("selected workspace motion shape is invalid"));
        }
        check_gravity(gravity)?;
        for i in 0..batch {
            self.robot
                .check_cmtm_motion(&motion[i * input_len..(i + 1) * input_len], self.order)?;
        }
        let dynamic = outputs.iter().any(|x| x.2 < 3);
        let replace = self.primal.len() != batch || self.dynamic != dynamic;
        if replace {
            self.primal = (0..batch)
                .map(|_| {
                    if dynamic {
                        DynamicsCmtmWorkspace::new(&self.robot, self.order - 2)
                    } else {
                        DynamicsCmtmWorkspace::kinematics_only(&self.robot, self.order)
                    }
                })
                .collect();
            self.tangent = None;
            self.ready = false;
        }
        let changed_gravity = dynamic && self.gravity != gravity;
        for i in 0..batch {
            let sample = &motion[i * input_len..(i + 1) * input_len];
            if !self.ready
                || changed_gravity
                || self.motion[i * input_len..(i + 1) * input_len] != *sample
            {
                if dynamic {
                    self.robot.dynamics_cmtm_into(
                        sample,
                        self.order - 2,
                        gravity,
                        &mut self.primal[i],
                    );
                    self.dynamics_evaluations += 1;
                } else {
                    self.robot
                        .kinematics_cmtm_into(sample, self.order, &mut self.primal[i].cmtm);
                    self.kinematics_evaluations += 1;
                }
            }
        }
        self.motion.clear();
        self.motion.extend_from_slice(motion);
        self.gravity = gravity;
        self.dynamic = dynamic;
        self.ready = true;
        Ok(())
    }

    /// Matrix-free selected JVP/VJP. Motion is (batch, dof * order), RHS is
    /// (batch, input-or-output rows, cols), and results are flattened row-major.
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &mut self,
        motion: &[f64],
        rhs: &[f64],
        outputs: &[StateOutput],
        batch: usize,
        cols: usize,
        gravity: [f64; 3],
        transpose: bool,
    ) -> CoreResult<Vec<f64>> {
        let packed = outputs.iter().map(StateOutput::raw).collect::<Vec<_>>();
        self.apply_raw(motion, rhs, &packed, batch, cols, gravity, transpose)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_raw(
        &mut self,
        motion: &[f64],
        rhs: &[f64],
        outputs: &[DynamicsOutput],
        batch: usize,
        cols: usize,
        gravity: [f64; 3],
        transpose: bool,
    ) -> CoreResult<Vec<f64>> {
        let rows = self.robot.check_selected_outputs(&outputs, self.order)?;
        let input_len = self.robot.dof * self.order;
        let rhs_rows = if transpose { rows } else { input_len };
        let out_rows = if transpose { input_len } else { rows };
        if motion.len() != checked_size(&[batch, input_len])?
            || rhs.len() != checked_size(&[batch, rhs_rows, cols])?
        {
            return Err(Error::new(
                "selected workspace motion or RHS shape is invalid",
            ));
        }
        checked_size(&[batch, out_rows, cols])?;
        self.prepare_selected_primal(motion, outputs, batch, gravity)?;
        let dynamic = outputs.iter().any(|x| x.2 < 3);
        if !transpose && self.tangent.as_ref().map(|x| x.rhs_cols) != Some(cols) {
            self.tangent = Some(if dynamic {
                DynamicsCmtmTangentWorkspace::new(&self.robot, self.order - 2, cols)
            } else {
                DynamicsCmtmTangentWorkspace::kinematics_only(&self.robot, self.order, cols)
            });
        }
        let mut out = vec![0.0; batch * out_rows * cols];
        for i in 0..batch {
            let sample = &motion[i * input_len..(i + 1) * input_len];
            let directions = &rhs[i * rhs_rows * cols..(i + 1) * rhs_rows * cols];
            let result = &mut out[i * out_rows * cols..(i + 1) * out_rows * cols];
            if transpose {
                self.robot.selected_reverse_from_state_into(
                    sample,
                    directions,
                    &outputs,
                    self.order,
                    gravity,
                    cols,
                    &mut self.primal[i],
                    result,
                );
            } else {
                self.robot.selected_tangent_from_state_into(
                    sample,
                    directions,
                    &outputs,
                    self.order,
                    gravity,
                    &mut self.primal[i],
                    self.tangent.as_mut().unwrap(),
                    result,
                );
            }
        }
        Ok(out)
    }
    pub fn cache_info(&self) -> (usize, usize, usize) {
        (
            self.kinematics_evaluations,
            self.dynamics_evaluations,
            self.primal.len(),
        )
    }
}

fn checked_size(dimensions: &[usize]) -> CoreResult<usize> {
    dimensions.iter().try_fold(1usize, |size, &dim| {
        size.checked_mul(dim)
            .ok_or_else(|| Error::new("array dimensions overflow"))
    })
}

fn check_gravity(gravity: [f64; 3]) -> CoreResult<()> {
    if gravity.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Error::new("gravity must contain only finite values"))
    }
}

impl RustAbaData {
    pub fn acceleration(&self) -> &[f64] {
        &self.workspace.qdd
    }
}

impl RustCompiledRobot {
    /// RNEA joint torques. Inputs are in DOF order; gravity is in world coordinates.
    /// Supports fixed/revolute/prismatic joints and does not require CMTM support.
    pub fn inverse_dynamics(
        &self,
        q: &[f64],
        v: &[f64],
        a: &[f64],
        gravity: [f64; 3],
    ) -> CoreResult<Vec<f64>> {
        self.check_motion(q, v, a)?;
        check_finite_motion(q, v, a)?;
        check_gravity(gravity)?;
        let mut ws = crate::workspace::Workspace::new(self);
        self.rnea_with_gravity_into(q, v, a, gravity, &mut ws);
        Ok(ws.tau)
    }

    /// ABA joint accelerations. Use create_aba_data for repeated cached solves.
    pub fn forward_dynamics(
        &self,
        q: &[f64],
        v: &[f64],
        tau: &[f64],
        gravity: [f64; 3],
    ) -> CoreResult<Vec<f64>> {
        self.check_motion(q, v, tau)?;
        check_finite_motion(q, v, tau)?;
        check_gravity(gravity)?;
        let mut ws = crate::workspace::AbaWorkspace::new(self);
        self.aba_with_gravity_into(q, v, tau, gravity, &mut ws)
            .map_err(Error::new)?;
        Ok(ws.qdd)
    }

    /// Flattened row-major (batch, dof) RNEA, sharing one temporary workspace.
    pub fn inverse_dynamics_batch(
        &self,
        q: &[f64],
        v: &[f64],
        a: &[f64],
        batch: usize,
        gravity: [f64; 3],
    ) -> CoreResult<Vec<f64>> {
        self.check_dynamics_batch_input(q, v, a, batch, gravity)?;
        let mut ws = crate::workspace::Workspace::new(self);
        let mut out = vec![0.; q.len()];
        for i in 0..batch {
            let range = i * self.dof..(i + 1) * self.dof;
            self.rnea_with_gravity_into(
                &q[range.clone()],
                &v[range.clone()],
                &a[range.clone()],
                gravity,
                &mut ws,
            );
            out[range].copy_from_slice(&ws.tau);
        }
        Ok(out)
    }

    /// Flattened row-major (batch, dof) ABA, sharing one temporary workspace.
    pub fn forward_dynamics_batch(
        &self,
        q: &[f64],
        v: &[f64],
        tau: &[f64],
        batch: usize,
        gravity: [f64; 3],
    ) -> CoreResult<Vec<f64>> {
        self.check_dynamics_batch_input(q, v, tau, batch, gravity)?;
        let mut ws = crate::workspace::AbaWorkspace::new(self);
        let mut out = vec![0.; q.len()];
        for i in 0..batch {
            let range = i * self.dof..(i + 1) * self.dof;
            self.aba_with_gravity_into(
                &q[range.clone()],
                &v[range.clone()],
                &tau[range.clone()],
                gravity,
                &mut ws,
            )
            .map_err(Error::new)?;
            out[range].copy_from_slice(&ws.qdd);
        }
        Ok(out)
    }

    fn check_dynamics_batch_input(
        &self,
        q: &[f64],
        v: &[f64],
        rhs: &[f64],
        batch: usize,
        gravity: [f64; 3],
    ) -> CoreResult<()> {
        let size = checked_size(&[batch, self.dof])?;
        if q.len() != size || v.len() != size || rhs.len() != size {
            return Err(Error::new("batch inputs must have length batch * dof"));
        }
        check_finite_motion(q, v, rhs)?;
        check_gravity(gravity)
    }
}

fn check_finite_motion(q: &[f64], v: &[f64], rhs: &[f64]) -> CoreResult<()> {
    if q.iter().chain(v).chain(rhs).all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Error::new(
            "motion and dynamics inputs must contain only finite values",
        ))
    }
}
