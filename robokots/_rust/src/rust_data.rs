use crate::error::{Error, CoreResult};

use crate::types::{RustBatchOutwardData, RustOutwardData};
use crate::workspace::CmtmWorkspace;

impl RustOutwardData {
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
