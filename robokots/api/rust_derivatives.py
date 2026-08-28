"""Rust-backed derivative kernels, including CMTM and kinetic energy."""
from __future__ import annotations

import numpy as np

from ..core import batch as batch_api
from ..core.state import StateType, keys_force, keys_kinematics, keys_momentum, keys_torque


class RustDerivativesMixin:
  def _rust_torque_row_parts(self, state_type_list, max_order : int):
    if self.dim_ != 3 or max_order != 3:
      return None
    rows = []
    part_sizes = []
    for st in state_type_list:
      if st.owner_type != "joint" or st.data_type != "torque" or st.frame_name is not None:
        return None
      joint = self.robot_.joint(st.owner_name)
      if joint is None or joint.dof <= 0:
        return None
      rows.extend(range(joint.dof_index, joint.dof_index + joint.dof))
      part_sizes.append(joint.dof)
    return rows, part_sizes

  def _rust_qva_order3(self):
    motion = np.asarray(self.motion(3), dtype=float)
    if motion.shape[-1] != self.robot_.dof * 3:
      return None
    if motion.ndim == 1:
      return (
        np.ascontiguousarray(motion[0::3]),
        np.ascontiguousarray(motion[1::3]),
        np.ascontiguousarray(motion[2::3]),
        (),
      )
    batch_shape = motion.shape[:-1]
    flat = motion.reshape((-1, motion.shape[-1]))
    return (
      np.ascontiguousarray(flat[:, 0::3]),
      np.ascontiguousarray(flat[:, 1::3]),
      np.ascontiguousarray(flat[:, 2::3]),
      batch_shape,
    )

  def _rust_torque_jacobian(self, state_type_list, max_order : int, list_output : bool = False):
    if not hasattr(self.outward_state_, "raw_data"):
      return None
    spec = self._rust_torque_row_parts(state_type_list, max_order)
    if spec is None:
      return None
    rows, part_sizes = spec
    qva = self._rust_qva_order3()
    if qva is None:
      return None
    q, v, a, batch_shape = qva
    try:
      if batch_shape:
        jacob = np.asarray(self._rust_compiled_robot().dynamics_jacobian_batch(q, v, a, gravity=self.gravity_))
        jacob = jacob.reshape(batch_shape + jacob.shape[-2:])
      else:
        jacob = np.asarray(self._rust_compiled_robot().dynamics_jacobian(q, v, a, gravity=self.gravity_))
    except Exception:
      return None
    selected = jacob[..., rows, :]
    if not list_output:
      return selected
    parts = []
    start = 0
    for size in part_sizes:
      parts.append(selected[..., start:start + size, :])
      start += size
    return parts

  def _rust_torque_jacobian_apply(self, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool, list_output : bool = False):
    spec = self._rust_torque_row_parts(state_type_list, max_order)
    if spec is None or not hasattr(self.outward_state_, "raw_data"):
      return None
    rows, part_sizes = spec
    qva = self._rust_qva_order3()
    if qva is None:
      return None
    q, v, a, motion_batch_shape = qva
    if tuple(batch_shape) != tuple(motion_batch_shape):
      return None
    try:
      if rhs_is_matrix:
        rhs_matrix = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
      else:
        rhs_vec = rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs
        rhs_matrix = rhs_vec[..., :, None]
      if batch_shape:
        flat_rhs = np.ascontiguousarray(rhs_matrix.reshape((-1,) + rhs_matrix.shape[-2:]))
        applied = np.asarray(self._rust_compiled_robot().dynamics_jacobian_matmul_rhs_batch(q, v, a, flat_rhs, gravity=self.gravity_))
        applied = applied.reshape(batch_shape + applied.shape[-2:])
      else:
        applied = np.asarray(self._rust_compiled_robot().dynamics_jacobian_matmul_rhs(q, v, a, np.ascontiguousarray(rhs_matrix), gravity=self.gravity_))
    except Exception:
      return None
    selected = applied[..., rows, :]

    if not rhs_is_matrix:
      selected = selected[..., 0]

    if list_output:
      parts = []
      start = 0
      for size in part_sizes:
        parts.append(selected[..., start:start + size] if not rhs_is_matrix else selected[..., start:start + size, :])
        start += size
      return parts
    return selected

  def _rust_cmtm_world_link_dynamics_jacobian_transpose_apply(self, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool):
    if not hasattr(self.outward_state_, "raw_data") or self.dim_ != 3 or max_order < 3:
      return None
    dynamics_order = max_order - 2
    link_ids = {link.name: i for i, link in enumerate(self.robot_.links)}
    parts = []
    for st in state_type_list:
      time = getattr(st, "key_order", 0) - 1
      if st.frame_name != "world" or st.owner_type != "link" or st.owner_name not in link_ids:
        return None
      if st.data_type in keys_momentum and 0 <= time <= dynamics_order:
        parts.append(("momentum", link_ids[st.owner_name], time))
      elif st.data_type in keys_force and 0 <= time < dynamics_order:
        parts.append(("force", link_ids[st.owner_name], time))
      else:
        return None
    motion = np.asarray(self.motion(max_order), dtype=float)
    if motion.shape[-1] != self.robot_.dof * max_order:
      return None
    try:
      rhs_matrix = rhs.reshape(batch_shape + rhs.shape[-2:]) if rhs_is_matrix and batch_shape else rhs if rhs_is_matrix else (rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs)[..., None]
      cols = rhs_matrix.shape[-1]
      base = rhs_matrix.shape[:-2]
      lm = np.zeros(base + (len(self.robot_.links), dynamics_order + 1, 6, cols))
      lf = np.zeros(base + (len(self.robot_.links), dynamics_order, 6, cols))
      row = 0
      for family, link, time in parts:
        (lm if family == "momentum" else lf)[..., link, time, :, :] += rhs_matrix[..., row:row + 6, :]
        row += 6
      robot = self._rust_compiled_robot()
      if batch_shape:
        flat_motion = motion.reshape((-1, motion.shape[-1])); flat_lm = lm.reshape((-1,) + lm.shape[len(batch_shape):]); flat_lf = lf.reshape((-1,) + lf.shape[len(batch_shape):])
        batch_kernel = getattr(robot, "world_link_dynamics_cmtm_transpose_matmul_rhs_batch", None)
        if batch_kernel is not None:
          out = np.asarray(batch_kernel(np.ascontiguousarray(flat_motion), np.ascontiguousarray(flat_lm), np.ascontiguousarray(flat_lf), dynamics_order, gravity=self.gravity_)).reshape(batch_shape + (motion.shape[-1], cols))
        else:
          out = np.stack([np.asarray(robot.world_link_dynamics_cmtm_transpose_matmul_rhs(m, a, b, dynamics_order, gravity=self.gravity_)) for m, a, b in zip(flat_motion, flat_lm, flat_lf)]).reshape(batch_shape + (motion.shape[-1], cols))
      else:
        out = np.asarray(robot.world_link_dynamics_cmtm_transpose_matmul_rhs(motion, lm, lf, dynamics_order, gravity=self.gravity_))
    except Exception:
      return None
    return out if rhs_is_matrix else out[..., 0]
  def _rust_cmtm_world_joint_dynamics_jacobian_transpose_apply(self, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool):
    if not hasattr(self.outward_state_, "raw_data") or self.dim_ != 3 or max_order < 3:
      return None
    dynamics_order = max_order - 2
    joint_ids = {joint.name: i for i, joint in enumerate(self.robot_.joints)}
    parts = []
    for st in state_type_list:
      time = getattr(st, "key_order", 0) - 1
      if st.frame_name != "world" or st.owner_type != "joint" or st.owner_name not in joint_ids:
        return None
      if st.data_type in keys_momentum and 0 <= time <= dynamics_order:
        parts.append(("momentum", joint_ids[st.owner_name], time))
      elif st.data_type in keys_force and 0 <= time < dynamics_order:
        parts.append(("force", joint_ids[st.owner_name], time))
      else:
        return None
    motion = np.asarray(self.motion(max_order), dtype=float)
    if motion.shape[-1] != self.robot_.dof * max_order:
      return None
    try:
      rhs_matrix = rhs.reshape(batch_shape + rhs.shape[-2:]) if rhs_is_matrix and batch_shape else rhs if rhs_is_matrix else (rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs)[..., None]
      cols = rhs_matrix.shape[-1]; base = rhs_matrix.shape[:-2]
      jm = np.zeros(base + (len(self.robot_.joints), dynamics_order + 1, 6, cols))
      jf = np.zeros(base + (len(self.robot_.joints), dynamics_order, 6, cols))
      row = 0
      for family, joint, time in parts:
        (jm if family == "momentum" else jf)[..., joint, time, :, :] += rhs_matrix[..., row:row + 6, :]
        row += 6
      robot = self._rust_compiled_robot()
      if batch_shape:
        flat_motion = motion.reshape((-1, motion.shape[-1])); flat_jm = jm.reshape((-1,) + jm.shape[len(batch_shape):]); flat_jf = jf.reshape((-1,) + jf.shape[len(batch_shape):])
        batch_kernel = getattr(robot, "world_joint_dynamics_cmtm_transpose_matmul_rhs_batch", None)
        if batch_kernel is not None:
          out = np.asarray(batch_kernel(np.ascontiguousarray(flat_motion), np.ascontiguousarray(flat_jm), np.ascontiguousarray(flat_jf), dynamics_order, gravity=self.gravity_)).reshape(batch_shape + (motion.shape[-1], cols))
        else:
          out = np.stack([np.asarray(robot.world_joint_dynamics_cmtm_transpose_matmul_rhs(m, a, b, dynamics_order, gravity=self.gravity_)) for m, a, b in zip(flat_motion, flat_jm, flat_jf)]).reshape(batch_shape + (motion.shape[-1], cols))
      else:
        out = np.asarray(robot.world_joint_dynamics_cmtm_transpose_matmul_rhs(motion, jm, jf, dynamics_order, gravity=self.gravity_))
    except Exception:
      return None
    return out if rhs_is_matrix else out[..., 0]

  def _rust_cmtm_jacobian_transpose_apply(self, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool):
    """Compose local CMTM kinematics and dynamics VJPs for mixed outputs.

    The two Rust kernels intentionally have compact, domain-specific output
    layouts.  This adapter preserves user StateType ordering, sends each
    subset to its native kernel, and sums their input cotangents.  Unsupported
    entries (world-frame/pose) return ``None`` atomically so the established
    complete outward reverse path remains responsible for the whole request.
    """
    kinetic = []
    dynamic = []
    widths = []
    for index, st in enumerate(state_type_list):
      kin_part = self._rust_cmtm_kinematics_row_parts([st], max_order)
      dyn_part = self._rust_cmtm_outward_dynamics_row_parts([st], max_order)
      if kin_part is not None:
        kinetic.append(index)
        widths.append(6)
      elif dyn_part is not None:
        dynamic.append(index)
        widths.append(sum(part[3] for part in dyn_part[1]))
      else:
        return None
    # Leave homogeneous requests to the specialised implementations below.
    # This avoids an otherwise unnecessary slicing/allocation on the common
    # single-family hot path.
    if not kinetic or not dynamic:
      return None
    try:
      rhs_matrix = rhs.reshape(batch_shape + rhs.shape[-2:]) if rhs_is_matrix and batch_shape else rhs if rhs_is_matrix else (rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs)[..., None]
      rhs_cols = rhs_matrix.shape[-1]
      row_start = 0
      kin_rows = []
      dyn_rows = []
      for index, width in enumerate(widths):
        target = kin_rows if index in kinetic else dyn_rows
        target.extend(range(row_start, row_start + width))
        row_start += width
      kin_rhs = np.take(rhs_matrix, kin_rows, axis=-2)
      dyn_rhs = np.take(rhs_matrix, dyn_rows, axis=-2)
      kin_states = [state_type_list[index] for index in kinetic]
      dyn_states = [state_type_list[index] for index in dynamic]
      kin_out = self._rust_cmtm_kinematics_jacobian_transpose_apply(
        kin_states, max_order, kin_rhs, batch_shape, True,
      )
      dyn_out = self._rust_cmtm_outward_dynamics_jacobian_transpose_apply(
        dyn_states, max_order, dyn_rhs, batch_shape, True,
      )
      if kin_out is None or dyn_out is None:
        return None
      out = kin_out + dyn_out
      return out if rhs_is_matrix else out[..., 0]
    except Exception:
      return None

  def _rust_cmtm_kinematics_row_parts(self, state_type_list, max_order : int):
    if self.dim_ != 3 or max_order < 3:
      return None
    link_ids = {link.name: link_id for link_id, link in enumerate(self.robot_.links)}
    joint_ids = {joint.name: joint_id for joint_id, joint in enumerate(self.robot_.joints)}
    parts = []
    for st in state_type_list:
      # Position/rotation/frame outputs include the CMTM transform and keep
      # their established path. Spatial derivative states are link/joint CMTM
      # vector entries and can use this direct kernel.
      if st.data_type not in keys_kinematics or st.frame_name is not None:
        return None
      time = getattr(st, "key_order", 0) - 2
      owner_ids = link_ids if st.owner_type == "link" else joint_ids if st.owner_type == "joint" else None
      if owner_ids is None or st.owner_name not in owner_ids or time < 0 or time >= max_order - 1:
        return None
      parts.append(("link" if st.owner_type == "link" else "joint", owner_ids[st.owner_name], time))
    return parts

  def _rust_cmtm_kinematics_jacobian_transpose_apply(self, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool):
    if not hasattr(self.outward_state_, "raw_data"):
      return None
    parts = self._rust_cmtm_kinematics_row_parts(state_type_list, max_order)
    if parts is None:
      return None
    motion = np.asarray(self.motion(max_order), dtype=float)
    motion_batch_shape = motion.shape[:-1] if motion.ndim > 1 else ()
    if motion.shape[-1] != self.robot_.dof * max_order or tuple(batch_shape) != tuple(motion_batch_shape):
      return None
    try:
      rhs_matrix = rhs.reshape(batch_shape + rhs.shape[-2:]) if rhs_is_matrix and batch_shape else rhs if rhs_is_matrix else (rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs)[..., None]
      rhs_cols = rhs_matrix.shape[-1]
      base_shape = rhs_matrix.shape[:-2]
      link_rhs = np.zeros(base_shape + (len(self.robot_.links), max_order - 1, 6, rhs_cols))
      joint_rhs = np.zeros(base_shape + (len(self.robot_.joints), max_order - 1, 6, rhs_cols))
      row = 0
      for owner_type, owner_id, time in parts:
        target = link_rhs if owner_type == "link" else joint_rhs
        target[..., owner_id, time, :, :] += rhs_matrix[..., row:row + 6, :]
        row += 6
      robot = self._rust_compiled_robot()
      if batch_shape:
        flat_motion = motion.reshape((-1, motion.shape[-1]))
        flat_link_rhs = link_rhs.reshape((-1,) + link_rhs.shape[len(batch_shape):])
        flat_joint_rhs = joint_rhs.reshape((-1,) + joint_rhs.shape[len(batch_shape):])
        out = np.stack([
          np.asarray(robot.cmtm_outward_kinematics_transpose_matmul_rhs(
            np.ascontiguousarray(sample_motion),
            np.ascontiguousarray(flat_link_rhs[i]),
            np.ascontiguousarray(flat_joint_rhs[i]),
            max_order,
          )) for i, sample_motion in enumerate(flat_motion)
        ]).reshape(batch_shape + (motion.shape[-1], rhs_cols))
      else:
        out = np.asarray(robot.cmtm_outward_kinematics_transpose_matmul_rhs(
          np.ascontiguousarray(motion), np.ascontiguousarray(link_rhs),
          np.ascontiguousarray(joint_rhs), max_order,
        ))
    except Exception:
      return None
    return out if rhs_is_matrix else out[..., 0]

  def _rust_cmtm_outward_dynamics_row_parts(self, state_type_list, max_order : int):
    """Map local high-order dynamics StateTypes to Rust CMTM output rows.

    World-frame values intentionally stay on the established Python reverse
    path: their frame transform is a separate differentiable operation.  All
    local momentum/force/wrench and torque states can be packed directly into
    the single Rust outward-dynamics VJP without materialising a Jacobian.
    """
    if self.dim_ != 3 or max_order < 3:
      return None
    dynamics_order = max_order - 2
    active = [(joint_id, joint) for joint_id, joint in enumerate(self.robot_.joints) if joint.dof > 0]
    if any(joint.dof != 1 for _, joint in active):
      return None
    link_ids = {link.name: link_id for link_id, link in enumerate(self.robot_.links)}
    joint_ids = {joint.name: joint_id for joint_id, joint in enumerate(self.robot_.joints)}
    parts = []
    for st in state_type_list:
      if st.frame_name is not None:
        return None
      time = getattr(st, "key_order", 0) - 1
      if st.data_type in keys_momentum:
        owner_ids = link_ids if st.owner_type == "link" else joint_ids if st.owner_type == "joint" else None
        if owner_ids is None or st.owner_name not in owner_ids or time < 0 or time > dynamics_order:
          return None
        parts.append(("link_momentum" if st.owner_type == "link" else "joint_momentum", owner_ids[st.owner_name], time, 6))
      elif st.data_type in keys_force:
        owner_ids = link_ids if st.owner_type == "link" else joint_ids if st.owner_type == "joint" else None
        if owner_ids is None or st.owner_name not in owner_ids or time < 0 or time >= dynamics_order:
          return None
        parts.append(("link_force" if st.owner_type == "link" else "joint_force", owner_ids[st.owner_name], time, 6))
      elif st.data_type in keys_torque:
        if st.owner_type == "joint":
          if st.owner_name not in joint_ids or time < 0 or time >= dynamics_order:
            return None
          parts.append(("joint_torque", joint_ids[st.owner_name], time, 1))
        elif st.owner_type == "total_joint" and st.owner_name == "total_joint" and 0 <= time < dynamics_order:
          for joint_id, _ in sorted(active, key=lambda item: item[1].dof_index):
            parts.append(("joint_torque", joint_id, time, 1))
        else:
          return None
      else:
        return None
    return dynamics_order, parts

  def _rust_cmtm_outward_dynamics_jacobian_transpose_apply(self, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool):
    if not hasattr(self.outward_state_, "raw_data"):
      return None
    spec = self._rust_cmtm_outward_dynamics_row_parts(state_type_list, max_order)
    if spec is None:
      return None
    dynamics_order, parts = spec
    motion = np.asarray(self.motion(max_order), dtype=float)
    motion_batch_shape = motion.shape[:-1] if motion.ndim > 1 else ()
    if motion.shape[-1] != self.robot_.dof * max_order or tuple(batch_shape) != tuple(motion_batch_shape):
      return None
    try:
      rhs_matrix = rhs.reshape(batch_shape + rhs.shape[-2:]) if rhs_is_matrix and batch_shape else rhs if rhs_is_matrix else (rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs)[..., None]
      rhs_cols = rhs_matrix.shape[-1]
      base_shape = rhs_matrix.shape[:-2]
      link_num, joint_num = len(self.robot_.links), len(self.robot_.joints)
      packed = {
        "link_momentum": np.zeros(base_shape + (link_num, dynamics_order + 1, 6, rhs_cols)),
        "link_force": np.zeros(base_shape + (link_num, dynamics_order, 6, rhs_cols)),
        "joint_momentum": np.zeros(base_shape + (joint_num, dynamics_order + 1, 6, rhs_cols)),
        "joint_force": np.zeros(base_shape + (joint_num, dynamics_order, 6, rhs_cols)),
        "joint_torque": np.zeros(base_shape + (joint_num, dynamics_order, rhs_cols)),
      }
      row = 0
      for family, owner_id, time, width in parts:
        if width == 1:
          packed[family][..., owner_id, time, :] += rhs_matrix[..., row, :]
        else:
          packed[family][..., owner_id, time, :, :] += rhs_matrix[..., row:row + width, :]
        row += width
      robot = self._rust_compiled_robot()
      def apply_one(sample_motion, sample_index=()):
        return np.asarray(robot.dynamics_cmtm_transpose_matmul_rhs(
          np.ascontiguousarray(sample_motion),
          np.ascontiguousarray(packed["link_momentum"][sample_index]),
          np.ascontiguousarray(packed["link_force"][sample_index]),
          np.ascontiguousarray(packed["joint_momentum"][sample_index]),
          np.ascontiguousarray(packed["joint_force"][sample_index]),
          np.ascontiguousarray(packed["joint_torque"][sample_index]),
          dynamics_order, gravity=self.gravity_,
        ))
      if batch_shape:
        flat_motion = motion.reshape((-1, motion.shape[-1]))
        flat_packed = {
          key: value.reshape((-1,) + value.shape[len(batch_shape):])
          for key, value in packed.items()
        }
        batch_kernel = getattr(robot, "dynamics_cmtm_transpose_matmul_rhs_batch", None)
        if batch_kernel is not None:
          out = np.asarray(batch_kernel(
            np.ascontiguousarray(flat_motion),
            np.ascontiguousarray(flat_packed["link_momentum"]),
            np.ascontiguousarray(flat_packed["link_force"]),
            np.ascontiguousarray(flat_packed["joint_momentum"]),
            np.ascontiguousarray(flat_packed["joint_force"]),
            np.ascontiguousarray(flat_packed["joint_torque"]),
            dynamics_order, gravity=self.gravity_,
          )).reshape(batch_shape + (motion.shape[-1], rhs_cols))
        else:
          out = np.stack([
            np.asarray(robot.dynamics_cmtm_transpose_matmul_rhs(
              np.ascontiguousarray(sample_motion),
              np.ascontiguousarray(flat_packed["link_momentum"][i]),
              np.ascontiguousarray(flat_packed["link_force"][i]),
              np.ascontiguousarray(flat_packed["joint_momentum"][i]),
              np.ascontiguousarray(flat_packed["joint_force"][i]),
              np.ascontiguousarray(flat_packed["joint_torque"][i]),
              dynamics_order, gravity=self.gravity_,
            )) for i, sample_motion in enumerate(flat_motion)
          ]).reshape(batch_shape + (motion.shape[-1], rhs_cols))
      else:
        out = apply_one(motion)
    except Exception:
      return None
    return out if rhs_is_matrix else out[..., 0]

  def _rust_cmtm_torque_row_parts(self, state_type_list, max_order : int):
    """Describe torque-series rows produced by the higher-order Rust CMTM API.

    The CMTM implementation currently represents one scalar coordinate per
    URDF joint.  Keep the guard deliberately narrow: unsupported joint models
    retain the established outward/CMTM Python implementation rather than
    silently selecting incorrectly laid-out rows.
    """
    if self.dim_ != 3 or max_order < 3:
      return None
    dynamics_order = max_order - 2
    active = [
      (joint_id, joint)
      for joint_id, joint in enumerate(self.robot_.joints)
      if joint.dof > 0
    ]
    if any(joint.dof != 1 for _, joint in active):
      return None

    parts = []
    for st in state_type_list:
      torque_order = getattr(st, "key_order", 0) - 1
      if st.data_type not in keys_torque or st.frame_name is not None or torque_order < 0 or torque_order >= dynamics_order:
        return None
      if st.owner_type == "joint":
        joint = self.robot_.joint(st.owner_name)
        if joint is None or joint.dof != 1:
          return None
        joint_id = next((i for i, candidate in enumerate(self.robot_.joints) if candidate is joint), None)
        if joint_id is None:
          return None
        parts.append(([(joint_id, torque_order)], 1))
      elif st.owner_type == "total_joint" and st.owner_name == "total_joint":
        # State vectors use dof_index order, which is not necessarily the
        # URDF joint-array order in models containing fixed joints.
        rows = [(joint_id, torque_order) for joint_id, joint in sorted(active, key=lambda item: item[1].dof_index)]
        parts.append((rows, len(rows)))
      else:
        return None
    return dynamics_order, parts

  def _rust_cmtm_torque_jacobian(self, state_type_list, max_order : int, list_output : bool = False):
    """Materialize higher-order torque Jacobians through the Rust CMTM Jv.

    This is only used by ``jacobian()``; product APIs retain the cheaper
    directional kernel.  A full input basis is sent as one RHS block, so a
    trajectory batch still crosses the Python/Rust boundary once.
    """
    if not hasattr(self.outward_state_, "raw_data"):
      return None
    spec = self._rust_cmtm_torque_row_parts(state_type_list, max_order)
    if spec is None:
      return None
    dynamics_order, part_specs = spec
    motion = np.asarray(self.motion(max_order), dtype=float)
    batch_shape = motion.shape[:-1] if motion.ndim > 1 else ()
    input_len = self.robot_.dof * max_order
    if motion.shape[-1] != input_len:
      return None
    robot = self._rust_compiled_robot()
    scalar_kernel = getattr(robot, "dynamics_joint_torque_series_tangent", None)
    batch_kernel = getattr(robot, "dynamics_joint_torque_series_tangent_batch", None)
    if scalar_kernel is None and batch_kernel is None:
      return None
    try:
      basis = np.eye(input_len)
      if batch_shape:
        flat_motion = np.ascontiguousarray(motion.reshape((-1, input_len)))
        flat_basis = np.broadcast_to(basis, (flat_motion.shape[0], input_len, input_len)).copy()
        if batch_kernel is not None:
          applied = np.asarray(batch_kernel(flat_motion, flat_basis, dynamics_order, gravity=self.gravity_))
        else:
          applied = np.stack([
            np.asarray(scalar_kernel(sample, basis, dynamics_order, gravity=self.gravity_))
            for sample in flat_motion
          ])
        applied = applied.reshape(batch_shape + applied.shape[-3:])
      else:
        applied = np.asarray(scalar_kernel(np.ascontiguousarray(motion), basis, dynamics_order, gravity=self.gravity_))
    except Exception:
      return None
    selected_parts = []
    for rows, _ in part_specs:
      selected_parts.append(np.stack(
        [applied[..., joint_id, time_order, :] for joint_id, time_order in rows], axis=-2,
      ))
    if list_output:
      return selected_parts
    return np.concatenate(selected_parts, axis=-2)

  def _rust_cmtm_torque_jacobian_apply(self, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool, list_output : bool = False):
    """Apply the analytic higher-order CMTM torque tangent without a Jacobian.

    The Rust API accepts one primal sample and a block of tangent columns.
    For a trajectory batch we retain those columns per frame and call the
    kernel once per frame; this still avoids Python CMTM construction and any
    dense Jacobian materialisation.
    """
    if not hasattr(self.outward_state_, "raw_data"):
      return None
    spec = self._rust_cmtm_torque_row_parts(state_type_list, max_order)
    if spec is None:
      return None
    dynamics_order, part_specs = spec
    motion = np.asarray(self.motion(max_order), dtype=float)
    motion_batch_shape = motion.shape[:-1] if motion.ndim > 1 else ()
    if motion.shape[-1] != self.robot_.dof * max_order or tuple(batch_shape) != tuple(motion_batch_shape):
      return None
    try:
      if rhs_is_matrix:
        rhs_matrix = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
      else:
        rhs_vec = rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs
        rhs_matrix = rhs_vec[..., :, None]
      robot = self._rust_compiled_robot()
      if batch_shape:
        flat_motion = np.ascontiguousarray(motion.reshape((-1, motion.shape[-1])))
        flat_rhs = np.ascontiguousarray(rhs_matrix.reshape((-1,) + rhs_matrix.shape[-2:]))
        batch_kernel = getattr(robot, "dynamics_joint_torque_series_tangent_batch", None)
        if batch_kernel is not None:
          applied = np.asarray(batch_kernel(flat_motion, flat_rhs, dynamics_order, gravity=self.gravity_))
        else:
          applied = np.stack([
            np.asarray(robot.dynamics_joint_torque_series_tangent(sample, direction, dynamics_order, gravity=self.gravity_))
            for sample, direction in zip(flat_motion, flat_rhs)
          ])
        applied = applied.reshape(batch_shape + applied.shape[-3:])
      else:
        applied = np.asarray(robot.dynamics_joint_torque_series_tangent(
          np.ascontiguousarray(motion), np.ascontiguousarray(rhs_matrix), dynamics_order, gravity=self.gravity_
        ))
    except Exception:
      return None

    parts = []
    for rows, size in part_specs:
      # Selecting each scalar row separately preserves the StateType order.
      selected = np.stack([applied[..., joint_id, time_order, :] for joint_id, time_order in rows], axis=-2)
      if not rhs_is_matrix:
        selected = selected[..., 0]
      parts.append(selected)
    if list_output:
      return parts
    return np.concatenate(parts, axis=-2 if rhs_is_matrix else -1)

  def _rust_cmtm_torque_jacobian_transpose_apply(self, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool):
    """Apply the higher-order CMTM torque VJP, without materialising ``J``.

    The Rust kernel takes cotangents in its native ``(URDF joint,
    time-order, rhs-column)`` layout.  StateType rows can be a subset or in a
    different order, so pack them here rather than making the backend depend
    on the Python state representation.  ``hasattr`` deliberately makes this
    a no-op while running against an older extension module; the established
    outward reverse path remains the compatibility fallback.
    """
    if not hasattr(self.outward_state_, "raw_data"):
      return None
    spec = self._rust_cmtm_torque_row_parts(state_type_list, max_order)
    if spec is None:
      return None
    dynamics_order, part_specs = spec
    motion = np.asarray(self.motion(max_order), dtype=float)
    motion_batch_shape = motion.shape[:-1] if motion.ndim > 1 else ()
    if motion.shape[-1] != self.robot_.dof * max_order or tuple(batch_shape) != tuple(motion_batch_shape):
      return None

    robot = self._rust_compiled_robot()
    kernel = getattr(robot, "dynamics_joint_torque_series_transpose_matmul_rhs", None)
    batch_kernel = getattr(robot, "dynamics_joint_torque_series_transpose_matmul_rhs_batch", None)
    if kernel is None and batch_kernel is None:
      return None

    try:
      if rhs_is_matrix:
        rhs_matrix = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
      else:
        rhs_vec = rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs
        rhs_matrix = rhs_vec[..., :, None]
      rhs_cols = rhs_matrix.shape[-1]
      packed = np.zeros(
        rhs_matrix.shape[:-2] + (len(self.robot_.joints), dynamics_order, rhs_cols),
        dtype=rhs_matrix.dtype,
      )
      row_start = 0
      for rows, size in part_specs:
        for local_row, (joint_id, time_order) in enumerate(rows):
          packed[..., joint_id, time_order, :] = rhs_matrix[..., row_start + local_row, :]
        row_start += size

      if batch_shape:
        flat_motion = np.ascontiguousarray(motion.reshape((-1, motion.shape[-1])))
        flat_packed = np.ascontiguousarray(packed.reshape((-1,) + packed.shape[-3:]))
        if batch_kernel is not None:
          out = np.asarray(batch_kernel(flat_motion, flat_packed, dynamics_order, gravity=self.gravity_))
        else:
          # Compatibility with the first scalar-only VJP implementation.  It
          # remains analytic and is replaced transparently when the batched
          # kernel is exported.
          out = np.stack([
            np.asarray(kernel(sample, cotangent, dynamics_order, gravity=self.gravity_))
            for sample, cotangent in zip(flat_motion, flat_packed)
          ])
        out = out.reshape(batch_shape + out.shape[-2:])
      else:
        if kernel is not None:
          out = np.asarray(kernel(
            np.ascontiguousarray(motion), np.ascontiguousarray(packed), dynamics_order,
            gravity=self.gravity_,
          ))
        else:
          # The batch export is also a valid scalar implementation contract.
          out = np.asarray(batch_kernel(
            np.ascontiguousarray(motion[None, :]),
            np.ascontiguousarray(packed[None, ...]),
            dynamics_order,
            gravity=self.gravity_,
          ))[0]
    except Exception:
      return None
    if rhs_is_matrix:
      return out
    return out[..., 0]

  def _rust_torque_jacobian_transpose_apply(self, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool):
    spec = self._rust_torque_row_parts(state_type_list, max_order)
    if spec is None or not hasattr(self.outward_state_, "raw_data"):
      return None
    rows, _ = spec
    qva = self._rust_qva_order3()
    if qva is None:
      return None
    q, v, a, motion_batch_shape = qva
    if tuple(batch_shape) != tuple(motion_batch_shape):
      return None
    try:
      if rhs_is_matrix:
        rhs_part = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
      else:
        rhs_vec = rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs
        rhs_part = rhs_vec[..., :, None]
      if len(rows) == self.robot_.dof and rows == list(range(self.robot_.dof)):
        full_rhs = rhs_part
      else:
        full_rhs = np.zeros(rhs_part.shape[:-2] + (self.robot_.dof, rhs_part.shape[-1]), dtype=rhs_part.dtype)
        full_rhs[..., rows, :] = rhs_part
      if batch_shape:
        flat_rhs = np.ascontiguousarray(full_rhs.reshape((-1,) + full_rhs.shape[-2:]))
        out = np.asarray(self._rust_compiled_robot().dynamics_jacobian_transpose_matmul_rhs_batch(q, v, a, flat_rhs, gravity=self.gravity_))
        out = out.reshape(batch_shape + out.shape[-2:])
      else:
        out = np.asarray(self._rust_compiled_robot().dynamics_jacobian_transpose_matmul_rhs(q, v, a, np.ascontiguousarray(full_rhs), gravity=self.gravity_))
    except Exception:
      return None
    if rhs_is_matrix:
      return out
    return out[..., 0]

  def _rust_link_local_specs(self, state_type_list, max_order : int):
    if self.dim_ != 3 or max_order != 3:
      return None
    code_map = {
      "vel": 0,
      "acc": 1,
      "momentum": 2,
      "momentum_diff1": 3,
      "force": 4,
    }
    link_ids = []
    data_codes = []
    for st in state_type_list:
      if st.owner_type != "link" or st.data_type not in code_map or st.frame_name is not None:
        return None
      link = self.robot_.link(st.owner_name)
      if link is None:
        return None
      link_ids.append(link.id)
      data_codes.append(code_map[st.data_type])
    return np.asarray(link_ids, dtype=np.int64), np.asarray(data_codes, dtype=np.int64)

  def _rust_link_local_jacobian(self, state_type_list, max_order : int, list_output : bool = False):
    if np.any(self.gravity_) or not hasattr(self.outward_state_, "raw_data"):
      return None
    spec = self._rust_link_local_specs(state_type_list, max_order)
    if spec is None:
      return None
    link_ids, data_codes = spec
    qva = self._rust_qva_order3()
    if qva is None:
      return None
    q, v, a, batch_shape = qva
    try:
      if batch_shape:
        parts = [
          np.asarray(self._rust_compiled_robot().link_local_jacobian(q[i], v[i], a[i], link_ids, data_codes))
          for i in range(q.shape[0])
        ]
        jacob = np.stack(parts, axis=0).reshape(batch_shape + parts[0].shape)
      else:
        jacob = np.asarray(self._rust_compiled_robot().link_local_jacobian(q, v, a, link_ids, data_codes))
    except Exception:
      return None
    if not list_output:
      return jacob
    return [jacob[..., i * 6:(i + 1) * 6, :] for i in range(len(link_ids))]

  def _rust_link_local_jacobian_apply(self, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool, list_output : bool = False):
    if np.any(self.gravity_) or not hasattr(self.outward_state_, "raw_data"):
      return None
    spec = self._rust_link_local_specs(state_type_list, max_order)
    if spec is None:
      return None
    link_ids, data_codes = spec
    qva = self._rust_qva_order3()
    if qva is None:
      return None
    q, v, a, motion_batch_shape = qva
    if tuple(batch_shape) != tuple(motion_batch_shape):
      return None
    try:
      if rhs_is_matrix:
        rhs_matrix = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
      else:
        rhs_vec = rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs
        rhs_matrix = rhs_vec[..., :, None]
      if batch_shape:
        flat_rhs = np.ascontiguousarray(rhs_matrix.reshape((-1,) + rhs_matrix.shape[-2:]))
        parts = [
          np.asarray(self._rust_compiled_robot().link_local_jacobian_matmul_rhs(q[i], v[i], a[i], flat_rhs[i], link_ids, data_codes))
          for i in range(q.shape[0])
        ]
        applied = np.stack(parts, axis=0).reshape(batch_shape + parts[0].shape)
      else:
        applied = np.asarray(self._rust_compiled_robot().link_local_jacobian_matmul_rhs(q, v, a, np.ascontiguousarray(rhs_matrix), link_ids, data_codes))
    except Exception:
      return None
    if not rhs_is_matrix:
      applied = applied[..., 0]
    if list_output:
      return [applied[..., i * 6:(i + 1) * 6] if not rhs_is_matrix else applied[..., i * 6:(i + 1) * 6, :] for i in range(len(link_ids))]
    return applied

  def _rust_link_local_jacobian_transpose_apply(self, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool):
    jacob = self._rust_link_local_jacobian(state_type_list, max_order, list_output=False)
    if jacob is None:
      return None
    jac_t = np.swapaxes(jacob, -1, -2)
    if rhs_is_matrix:
      rhs_part = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
      return jac_t @ rhs_part
    rhs_part = rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs
    return (jac_t @ rhs_part[..., None])[..., 0]

  def kinetic_energy_state(self):
    """Return total kinetic energy from the current joint coordinates and velocities.

    Energy depends only on ``motion(2)`` (`q`, `qdot`).  The non-batched
    result is a Python float; batched motions return an array with the motion
    batch shape.
    """
    motion = np.asarray(self.motion(2), dtype=float)
    batch_shape = motion.shape[:-1] if batch_api.is_batched_feature_array(motion) else ()
    robot = self._rust_compiled_robot()
    if not batch_shape:
      return float(np.asarray(robot.kinetic_energy(np.ascontiguousarray(motion)))[0])
    flat_motion = np.ascontiguousarray(motion.reshape((-1, motion.shape[-1])))
    return np.asarray(robot.kinetic_energy_batch(flat_motion)).reshape(batch_shape)

  def kinetic_energy_jacobian_mul(self, rhs : np.ndarray):
    """Apply the kinetic-energy Jacobian to q/qdot directions.

    ``rhs`` follows :meth:`jacobian_mul` conventions with input dimension
    ``2 * dof``.  The scalar energy output retains its row dimension of one.
    """
    motion = np.asarray(self.motion(2), dtype=float)
    batch_shape = motion.shape[:-1] if batch_api.is_batched_feature_array(motion) else ()
    input_dim = self.robot_.dof * 2
    rhs, rhs_is_matrix = batch_api.broadcast_feature_rhs(rhs, batch_shape, input_dim, name="rhs")
    tangent = rhs if rhs_is_matrix else rhs[..., None]
    robot = self._rust_compiled_robot()
    if not batch_shape:
      out = np.asarray(robot.kinetic_energy_jacobian_mul_rhs(
        np.ascontiguousarray(motion), np.ascontiguousarray(tangent),
      ))
    else:
      flat_motion = np.ascontiguousarray(motion.reshape((-1, motion.shape[-1])))
      out = np.asarray(robot.kinetic_energy_jacobian_mul_rhs_batch(
        flat_motion, np.ascontiguousarray(tangent),
      )).reshape(batch_shape + (1, tangent.shape[-1]))
    return out if rhs_is_matrix else out[..., 0]

  def kinetic_energy_jacobian_transpose_mul(self, rhs : np.ndarray):
    """Apply the kinetic-energy VJP to scalar output cotangents.

    ``rhs`` has one output row, with optional final RHS-column axis.  The
    returned gradient is ordered ``[q0, qdot0, q1, qdot1, ...]``.
    """
    motion = np.asarray(self.motion(2), dtype=float)
    batch_shape = motion.shape[:-1] if batch_api.is_batched_feature_array(motion) else ()
    rhs, rhs_is_matrix = batch_api.broadcast_feature_rhs(rhs, batch_shape, 1, name="rhs")
    cotangent = rhs if rhs_is_matrix else rhs[..., None]
    robot = self._rust_compiled_robot()
    if not batch_shape:
      out = np.asarray(robot.kinetic_energy_jacobian_transpose_mul_rhs(
        np.ascontiguousarray(motion), np.ascontiguousarray(cotangent),
      ))
    else:
      flat_motion = np.ascontiguousarray(motion.reshape((-1, motion.shape[-1])))
      out = np.asarray(robot.kinetic_energy_jacobian_transpose_mul_rhs_batch(
        flat_motion, np.ascontiguousarray(cotangent),
      )).reshape(batch_shape + (self.robot_.dof * 2, cotangent.shape[-1]))
    return out if rhs_is_matrix else out[..., 0]
