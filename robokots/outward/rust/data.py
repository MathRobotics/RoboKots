from __future__ import annotations

import numpy as np
from mathrobo import SE3

from ...core import batch as batch_api
from ...core.robot import RobotStruct
from ...core.state import keys_force, keys_kinematics, keys_momentum, keys_torque
from ...core.state_dict import cmtm_to_state_list, vecs_to_state_dict
from .model import _rust_compiled_robot

def _skew(v: np.ndarray) -> np.ndarray:
  v = np.asarray(v)
  out = np.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype)
  out[..., 0, 1] = -v[..., 2]
  out[..., 0, 2] = v[..., 1]
  out[..., 1, 0] = v[..., 2]
  out[..., 1, 2] = -v[..., 0]
  out[..., 2, 0] = -v[..., 1]
  out[..., 2, 1] = v[..., 0]
  return out

def _factorial(order: int, dtype) -> np.ndarray:
  fact = np.ones(order, dtype=dtype)
  for i in range(1, order):
    fact[i] = fact[i - 1] * i
  return fact

def _mat4_inv_se3(mat: np.ndarray) -> np.ndarray:
  mat = np.asarray(mat)
  rot_t = np.swapaxes(mat[..., :3, :3], -1, -2)
  pos = -(rot_t @ mat[..., :3, 3, None])[..., 0]
  out = np.zeros(mat.shape, dtype=mat.dtype)
  out[..., :3, :3] = rot_t
  out[..., :3, 3] = pos
  out[..., 3, 3] = 1.0
  return out

def _hat_se3(vec: np.ndarray) -> np.ndarray:
  vec = np.asarray(vec)
  out = np.zeros(vec.shape[:-1] + (4, 4), dtype=vec.dtype)
  out[..., :3, :3] = _skew(vec[..., :3])
  out[..., :3, 3] = vec[..., 3:6]
  return out

def _vee_se3(mat: np.ndarray) -> np.ndarray:
  mat = np.asarray(mat)
  out = np.empty(mat.shape[:-2] + (6,), dtype=mat.dtype)
  out[..., 0] = 0.5 * (mat[..., 2, 1] - mat[..., 1, 2])
  out[..., 1] = 0.5 * (mat[..., 0, 2] - mat[..., 2, 0])
  out[..., 2] = 0.5 * (mat[..., 1, 0] - mat[..., 0, 1])
  out[..., 3:6] = mat[..., :3, 3]
  return out

def _mat_adj_from_mat4(mat: np.ndarray, wrench: bool) -> np.ndarray:
  mat = np.asarray(mat)
  rot = mat[..., :3, :3]
  pos = mat[..., :3, 3]
  shifted = _skew(pos) @ rot
  out = np.zeros(mat.shape[:-2] + (6, 6), dtype=mat.dtype)
  out[..., :3, :3] = rot
  out[..., 3:6, 3:6] = rot
  if wrench:
    out[..., :3, 3:6] = shifted
  else:
    out[..., 3:6, :3] = shifted
  return out

def _mat_inv_adj_from_mat4(mat: np.ndarray, wrench: bool) -> np.ndarray:
  mat = np.asarray(mat)
  rot_t = np.swapaxes(mat[..., :3, :3], -1, -2)
  pos = mat[..., :3, 3]
  shifted = -(rot_t @ _skew(pos))
  out = np.zeros(mat.shape[:-2] + (6, 6), dtype=mat.dtype)
  out[..., :3, :3] = rot_t
  out[..., 3:6, 3:6] = rot_t
  if wrench:
    out[..., :3, 3:6] = shifted
  else:
    out[..., 3:6, :3] = shifted
  return out

def _hat_adj(vec: np.ndarray, wrench: bool) -> np.ndarray:
  vec = np.asarray(vec)
  w_hat = _skew(vec[..., :3])
  v_hat = _skew(vec[..., 3:6])
  out = np.zeros(vec.shape[:-1] + (6, 6), dtype=vec.dtype)
  out[..., :3, :3] = w_hat
  out[..., 3:6, 3:6] = w_hat
  if wrench:
    out[..., :3, 3:6] = v_hat
  else:
    out[..., 3:6, :3] = v_hat
  return out

def _hat_commute_adj(vec: np.ndarray, wrench: bool) -> np.ndarray:
  if not wrench:
    return -_hat_adj(vec, wrench=False)

  vec = np.asarray(vec)
  out = np.zeros(vec.shape[:-1] + (6, 6), dtype=vec.dtype)
  out[..., :3, :3] = _skew(vec[..., :3])
  out[..., :3, 3:6] = _skew(vec[..., 3:6])
  out[..., 3:6, :3] = _skew(vec[..., 3:6])
  return -out

def _flatten_last2(value: np.ndarray) -> np.ndarray:
  value = np.asarray(value)
  return value.reshape(value.shape[:-2] + (value.shape[-2] * value.shape[-1],))

def _lower_toeplitz(blocks: np.ndarray) -> np.ndarray:
  order = blocks.shape[-3]
  size = blocks.shape[-1]
  out = np.zeros(blocks.shape[:-3] + (order * size, order * size), dtype=blocks.dtype)
  for i in range(order):
    block = blocks[..., i, :, :]
    for j in range(i, order):
      out[..., size * j:size * (j + 1), size * (j - i):size * (j - i + 1)] = block
  return out

def _lower_tri_blocks(blocks: np.ndarray, col_scales: np.ndarray | None = None) -> np.ndarray:
  order = blocks.shape[-4]
  size = blocks.shape[-1]
  out = np.zeros(blocks.shape[:-4] + (order * size, order * size), dtype=blocks.dtype)
  for i in range(order):
    for j in range(i + 1):
      block = blocks[..., i, j, :, :]
      if col_scales is not None:
        block = block * col_scales[j]
      out[..., size * i:size * (i + 1), size * j:size * (j + 1)] = block
  return out

class _RustCMVectorView:
  """CMVector-compatible view over arrays owned by a Rust outward state."""

  def __init__(self, vecs: np.ndarray):
    self._vecs = np.asarray(vecs, dtype=float)
    self._n = self._vecs.shape[-2]
    self._dim = self._vecs.shape[-1]
    self._len = self._n * self._dim
    fact = _factorial(self._n, self._vecs.dtype)
    self._cm_vecs = self._vecs / fact.reshape((1,) * (self._vecs.ndim - 2) + (self._n, 1))

  @staticmethod
  def set_cmvecs(cm_vecs: np.ndarray) -> "_RustCMVectorView":
    cm_vecs = np.asarray(cm_vecs, dtype=float)
    fact = _factorial(cm_vecs.shape[-2], cm_vecs.dtype)
    return _RustCMVectorView(cm_vecs * fact.reshape((1,) * (cm_vecs.ndim - 2) + (cm_vecs.shape[-2], 1)))

  def vecs(self) -> np.ndarray:
    return self._vecs

  def cm_vecs(self) -> np.ndarray:
    return self._cm_vecs

  def vec(self) -> np.ndarray:
    return _flatten_last2(self._vecs)

  def cm_vec(self) -> np.ndarray:
    return _flatten_last2(self._cm_vecs)

  def truncate(self, order: int) -> "_RustCMVectorView":
    return _RustCMVectorView(self._vecs[..., :order, :])

def _as_rust_cmvector(value) -> _RustCMVectorView:
  if isinstance(value, _RustCMVectorView):
    return value
  if hasattr(value, "vecs"):
    return _RustCMVectorView(value.vecs())
  if hasattr(value, "cm_vecs"):
    return _RustCMVectorView.set_cmvecs(value.cm_vecs())
  return _RustCMVectorView(value)

def _hat_cm_commute_adj(vec: _RustCMVectorView, wrench: bool) -> np.ndarray:
  cm_vecs = vec.cm_vecs()
  order = vec._n
  size = vec._dim
  dtype = np.result_type(cm_vecs.dtype, np.longdouble)
  out = np.zeros(cm_vecs.shape[:-2] + (order * size, order * size), dtype=dtype)
  for i in range(order):
    block = _hat_commute_adj(cm_vecs[..., i, :], wrench).astype(dtype, copy=False)
    for j in range(order - i):
      out[..., size * (i + j):size * (i + j + 1), size * j:size * (j + 1)] = block
  return out

def _apply_or_transpose(mat: np.ndarray, vec: np.ndarray, transpose: bool) -> np.ndarray:
  mat = np.asarray(mat)
  vec = np.asarray(vec)
  if transpose:
    return (np.swapaxes(mat, -1, -2) @ vec[..., None])[..., 0]
  return (mat @ vec[..., None])[..., 0]

class _RustCMTMView:
  """CMTM-compatible view over Rust-produced matrix/vector arrays.

  This adapter is intentionally private. It exists so the existing jacobian and
  state_dict helpers can consume Rust-backed state without materializing
  mathrobo.CMTM objects eagerly.
  """

  def __init__(self, mat: np.ndarray, vecs: np.ndarray, wrench: bool = False, rust_wrench_kernel=None):
    self._mat = np.asarray(mat, dtype=float)
    self._vecs = np.asarray(vecs, dtype=float)
    self._n = self._vecs.shape[-2] + 1
    self._dof = 6
    self._mat_adj_size = 6
    self._wrench = bool(wrench)
    self._rust_wrench_kernel = rust_wrench_kernel
    self._cache = {}

  def elem_mat(self) -> np.ndarray:
    return self._mat

  def elem_vecs(self, i: int) -> np.ndarray:
    if i < 0 or i >= self._n - 1:
      return None
    return self._vecs[..., i, :]

  def vecs(self, output_order=None) -> np.ndarray:
    output_order = self._check_order(output_order)
    return self._vecs[..., : output_order - 1, :]

  def mat(self, output_order=None) -> np.ndarray:
    output_order = self._check_order(output_order)
    key = ("mat", output_order)
    if key not in self._cache:
      self._cache[key] = _lower_toeplitz(self._mat_blocks(output_order))
    return self._cache[key].copy()

  def mat_inv(self, output_order=None) -> np.ndarray:
    output_order = self._check_order(output_order)
    key = ("mat_inv", output_order)
    if key not in self._cache:
      self._cache[key] = _lower_toeplitz(self._mat_inv_blocks(output_order))
    return self._cache[key].copy()

  def mat_adj(self, output_order=None) -> np.ndarray:
    output_order = self._check_order(output_order)
    key = ("mat_adj", output_order, self._wrench)
    if key not in self._cache:
      self._cache[key] = _lower_toeplitz(self._mat_adj_blocks(output_order, inverse=False))
    return self._cache[key].copy()

  def mat_inv_adj(self, output_order=None) -> np.ndarray:
    output_order = self._check_order(output_order)
    key = ("mat_inv_adj", output_order, self._wrench)
    if key not in self._cache:
      self._cache[key] = _lower_toeplitz(self._mat_adj_blocks(output_order, inverse=True))
    return self._cache[key].copy()

  def tangent_mat(self, output_order=None) -> np.ndarray:
    output_order = self._check_order(output_order)
    key = ("tangent_mat", output_order)
    if key not in self._cache:
      table = self._tangent_table(output_order)
      scales = np.ones(output_order, dtype=table.dtype)
      fact = _factorial(output_order, table.dtype)
      for j in range(2, output_order):
        scales[j] = 1.0 / fact[j - 1]
      self._cache[key] = _lower_tri_blocks(table, col_scales=scales)
    return self._cache[key].copy()

  def tangent_mat_inv(self, output_order=None) -> np.ndarray:
    output_order = self._check_order(output_order)
    key = ("tangent_mat_inv", output_order)
    if key not in self._cache:
      self._cache[key] = np.linalg.inv(self.tangent_mat(output_order))
    return self._cache[key].copy()

  def inv(self) -> "_RustCMTMView":
    return _RustCMTMView._from_mat_blocks(
      self._mat_inv_blocks(self._n),
      wrench=self._wrench,
      rust_wrench_kernel=self._rust_wrench_kernel,
    )

  def as_wrench(self) -> "_RustCMTMView":
    return _RustCMTMView(self._mat, self._vecs, wrench=True, rust_wrench_kernel=self._rust_wrench_kernel)

  def cmvecs(self) -> _RustCMVectorView:
    return _RustCMVectorView(self._vecs)

  def mat_var_x_arb_vec_jacob(self, arb_vec, frame: str = "bframe") -> np.ndarray:
    arb_vec = _as_rust_cmvector(arb_vec)
    if frame == "bframe":
      return self.mat_adj() @ _hat_cm_commute_adj(arb_vec, self._wrench)
    if frame == "fframe":
      return _hat_cm_commute_adj(self @ arb_vec, self._wrench)
    raise ValueError("frame must be 'bframe' or 'fframe'")

  def mat_var_x_arb_vec_matvec(self, arb_vec, vec: np.ndarray, frame: str = "bframe", transpose: bool = False):
    if frame == "bframe":
      fast = self._rust_wrench_var_jacob_matvec(arb_vec, vec, inverse=False, transpose=transpose)
      if fast is not None:
        return fast
    return _apply_or_transpose(self.mat_var_x_arb_vec_jacob(arb_vec, frame=frame), vec, transpose)

  def mat_var_x_arb_vec_matmul_rhs(self, arb_vec, rhs: np.ndarray, frame: str = "bframe"):
    if frame == "bframe":
      fast = self._rust_wrench_var_jacob_matmul_rhs(arb_vec, rhs, inverse=False)
      if fast is not None:
        return fast
    return self.mat_var_x_arb_vec_jacob(arb_vec, frame=frame) @ rhs

  def mat_inv_var_x_arb_vec_jacob(self, arb_vec, frame: str = "bframe") -> np.ndarray:
    if frame != "bframe":
      raise NotImplementedError("Not implemented for fframe")
    return -_hat_cm_commute_adj(self.inv() @ _as_rust_cmvector(arb_vec), self._wrench)

  def mat_inv_var_x_arb_vec_matvec(self, arb_vec, vec: np.ndarray, frame: str = "bframe", transpose: bool = False):
    if frame == "bframe":
      fast = self._rust_wrench_var_jacob_matvec(arb_vec, vec, inverse=True, transpose=transpose)
      if fast is not None:
        return fast
    return _apply_or_transpose(self.mat_inv_var_x_arb_vec_jacob(arb_vec, frame=frame), vec, transpose)

  def mat_inv_var_x_arb_vec_matmul_rhs(self, arb_vec, rhs: np.ndarray, frame: str = "bframe"):
    if frame == "bframe":
      fast = self._rust_wrench_var_jacob_matmul_rhs(arb_vec, rhs, inverse=True)
      if fast is not None:
        return fast
    return self.mat_inv_var_x_arb_vec_jacob(arb_vec, frame=frame) @ rhs

  def __matmul__(self, other):
    if isinstance(other, _RustCMVectorView) or hasattr(other, "cm_vecs"):
      other = _as_rust_cmvector(other)
      if self._n != other._n:
        raise TypeError("Right operand should be same order")
      cm_vec = (self.mat_adj() @ other.cm_vec()[..., None])[..., 0]
      return _RustCMVectorView.set_cmvecs(cm_vec.reshape(cm_vec.shape[:-1] + (self._n, self._mat_adj_size)))
    if not isinstance(other, _RustCMTMView):
      return self.mat() @ other
    if self._n != other._n:
      raise TypeError("Right operand should be same order")
    l_blocks = self._mat_blocks(self._n)
    r_blocks = other._mat_blocks(other._n)
    out_blocks = np.zeros_like(l_blocks)
    for k in range(self._n):
      acc = np.zeros(l_blocks.shape[:-3] + (4, 4), dtype=np.result_type(l_blocks, r_blocks))
      for i in range(k + 1):
        acc = acc + l_blocks[..., i, :, :] @ r_blocks[..., k - i, :, :]
      out_blocks[..., k, :, :] = acc
    return _RustCMTMView._from_mat_blocks(
      out_blocks,
      wrench=self._wrench,
      rust_wrench_kernel=self._rust_wrench_kernel,
    )

  @staticmethod
  def _from_mat_blocks(
    blocks: np.ndarray,
    wrench: bool = False,
    rust_wrench_kernel=None,
  ) -> "_RustCMTMView":
    blocks = np.asarray(blocks, dtype=float)
    order = blocks.shape[-3]
    mat = blocks[..., 0, :, :]
    vecs = np.zeros(blocks.shape[:-3] + (max(order - 1, 0), 6), dtype=blocks.dtype)
    if order <= 1:
      return _RustCMTMView(mat, vecs, wrench=wrench, rust_wrench_kernel=rust_wrench_kernel)
    fact = _factorial(order, blocks.dtype)
    inv_mat = _mat4_inv_se3(mat)
    hats = []
    for i in range(order - 1):
      tmp = np.zeros(blocks.shape[:-3] + (4, 4), dtype=blocks.dtype)
      for j in range(i):
        tmp = tmp + blocks[..., i - j, :, :] @ hats[j]
      delta = inv_mat @ (blocks[..., i + 1, :, :] * (i + 1) - tmp)
      raw = _vee_se3(delta) * fact[i]
      vecs[..., i, :] = raw
      hats.append(_hat_se3(raw / fact[i]))
    return _RustCMTMView(mat, vecs, wrench=wrench, rust_wrench_kernel=rust_wrench_kernel)

  def _rust_wrench_var_jacob_matvec(
    self,
    arb_vec,
    vec: np.ndarray,
    inverse: bool,
    transpose: bool,
  ):
    if not self._wrench or self._rust_wrench_kernel is None:
      return None
    arb_vec = _as_rust_cmvector(arb_vec)
    vec = np.asarray(vec, dtype=float)
    if arb_vec._n != self._n or vec.shape[-1] != self._n * self._mat_adj_size:
      return None
    if vec.ndim == 1 and self._mat.ndim == 2 and self._vecs.ndim == 2 and arb_vec.cm_vecs().ndim == 2:
      return np.asarray(
        self._rust_wrench_kernel(
          np.ascontiguousarray(self._mat),
          np.ascontiguousarray(self._vecs),
          np.ascontiguousarray(arb_vec.cm_vecs()),
          np.ascontiguousarray(vec),
          bool(inverse),
          bool(transpose),
        )
      )
    if (
      vec.ndim == 2
      and self._mat.ndim == 3
      and self._vecs.ndim == 3
      and arb_vec.cm_vecs().ndim == 3
      and vec.shape[:-1] == self._mat.shape[:-2] == self._vecs.shape[:-2] == arb_vec.cm_vecs().shape[:-2]
    ):
      flat_shape = (-1, self._n * self._mat_adj_size)
      flat = np.asarray(
        self._rust_wrench_kernel(
          np.ascontiguousarray(self._mat.reshape((-1, 4, 4))),
          np.ascontiguousarray(self._vecs.reshape((-1, self._n - 1, 6))),
          np.ascontiguousarray(arb_vec.cm_vecs().reshape((-1, self._n, 6))),
          np.ascontiguousarray(vec.reshape(flat_shape)),
          bool(inverse),
          bool(transpose),
        )
      )
      return flat.reshape(vec.shape)
    return None

  def _rust_wrench_var_jacob_matmul_rhs(self, arb_vec, rhs: np.ndarray, inverse: bool):
    if not self._wrench or self._rust_wrench_kernel is None:
      return None
    kernel = getattr(self._rust_wrench_kernel.__self__, "cmtm_wrench_var_jacob_matmul_rhs", None)
    if kernel is None:
      return None
    arb_vec = _as_rust_cmvector(arb_vec)
    rhs = np.asarray(rhs, dtype=float)
    if arb_vec._n != self._n or rhs.ndim < 2 or rhs.shape[-2] != self._n * self._mat_adj_size:
      return None
    if rhs.ndim == 2 and self._mat.ndim == 2 and self._vecs.ndim == 2 and arb_vec.cm_vecs().ndim == 2:
      return np.asarray(
        kernel(
          np.ascontiguousarray(self._mat),
          np.ascontiguousarray(self._vecs),
          np.ascontiguousarray(arb_vec.cm_vecs()),
          np.ascontiguousarray(rhs),
          bool(inverse),
        )
      )
    if (
      rhs.ndim == 3
      and self._mat.ndim == 3
      and self._vecs.ndim == 3
      and arb_vec.cm_vecs().ndim == 3
      and rhs.shape[:-2] == self._mat.shape[:-2] == self._vecs.shape[:-2] == arb_vec.cm_vecs().shape[:-2]
    ):
      flat = np.asarray(
        kernel(
          np.ascontiguousarray(self._mat.reshape((-1, 4, 4))),
          np.ascontiguousarray(self._vecs.reshape((-1, self._n - 1, 6))),
          np.ascontiguousarray(arb_vec.cm_vecs().reshape((-1, self._n, 6))),
          np.ascontiguousarray(rhs.reshape((-1, self._n * self._mat_adj_size, rhs.shape[-1]))),
          bool(inverse),
        )
      )
      return flat.reshape(rhs.shape)
    return None

  def _check_order(self, output_order) -> int:
    if output_order is None:
      output_order = self._n
    output_order = int(output_order)
    if output_order < 0:
      output_order = self._n + output_order
    if output_order < 0 or output_order > self._n:
      raise TypeError("Output order should be less than or equal to the order of CMTM")
    return output_order

  def _mat_blocks(self, output_order: int) -> np.ndarray:
    key = ("mat_blocks", output_order)
    if key in self._cache:
      return self._cache[key]
    fact = _factorial(output_order, self._mat.dtype)
    blocks = np.zeros(self._mat.shape[:-2] + (output_order, 4, 4), dtype=self._mat.dtype)
    if output_order > 0:
      blocks[..., 0, :, :] = self._mat
    for k in range(1, output_order):
      acc = np.zeros(self._mat.shape[:-2] + (4, 4), dtype=self._mat.dtype)
      for i in range(k):
        acc = acc + blocks[..., k - i - 1, :, :] @ _hat_se3(self._vecs[..., i, :] / fact[i])
      blocks[..., k, :, :] = acc / k
    self._cache[key] = blocks
    return blocks

  def _mat_inv_blocks(self, output_order: int) -> np.ndarray:
    key = ("mat_inv_blocks", output_order)
    if key in self._cache:
      return self._cache[key]
    fact = _factorial(output_order, self._mat.dtype)
    blocks = np.zeros(self._mat.shape[:-2] + (output_order, 4, 4), dtype=self._mat.dtype)
    if output_order > 0:
      blocks[..., 0, :, :] = _mat4_inv_se3(self._mat)
    for k in range(1, output_order):
      acc = np.zeros(self._mat.shape[:-2] + (4, 4), dtype=self._mat.dtype)
      for i in range(k):
        acc = acc - _hat_se3(self._vecs[..., i, :] / fact[i]) @ blocks[..., k - i - 1, :, :]
      blocks[..., k, :, :] = acc / k
    self._cache[key] = blocks
    return blocks

  def _mat_adj_blocks(self, output_order: int, inverse: bool) -> np.ndarray:
    key = ("mat_adj_blocks", output_order, inverse, self._wrench)
    if key in self._cache:
      return self._cache[key]
    fact = _factorial(output_order, self._mat.dtype)
    blocks = np.zeros(self._mat.shape[:-2] + (output_order, 6, 6), dtype=self._mat.dtype)
    if output_order > 0:
      blocks[..., 0, :, :] = (
        _mat_inv_adj_from_mat4(self._mat, self._wrench)
        if inverse
        else _mat_adj_from_mat4(self._mat, self._wrench)
      )
    for k in range(1, output_order):
      acc = np.zeros(self._mat.shape[:-2] + (6, 6), dtype=self._mat.dtype)
      for i in range(k):
        hat = _hat_adj(self._vecs[..., i, :] / fact[i], self._wrench)
        if inverse:
          acc = acc - hat @ blocks[..., k - i - 1, :, :]
        else:
          acc = acc + blocks[..., k - i - 1, :, :] @ hat
      blocks[..., k, :, :] = acc / k
    self._cache[key] = blocks
    return blocks

  def _tangent_table(self, output_order: int) -> np.ndarray:
    key = ("tangent_table", output_order)
    if key in self._cache:
      return self._cache[key]
    table = np.zeros(self._mat.shape[:-2] + (output_order, output_order, 6, 6), dtype=self._mat.dtype)
    if output_order == 0:
      return table
    eye = np.eye(6, dtype=self._mat.dtype)
    table[..., 0, 0, :, :] = eye
    fact = _factorial(output_order, self._mat.dtype)
    for i in range(1, output_order):
      table[..., i, i, :, :] = eye / i
      for j in range(i):
        acc = np.zeros(self._mat.shape[:-2] + (6, 6), dtype=self._mat.dtype)
        for k in range(i - j):
          acc = acc - _hat_adj(self._vecs[..., k, :] / fact[k], self._wrench) @ table[..., i - k - 1, j, :, :]
        table[..., i, j, :, :] = acc / i
    self._cache[key] = table
    return table

class RustOutwardState:
  """Python outward-state adapter backed by raw PyO3 Rust workspace data.

  The raw_data object owns computation and storage. This class owns RoboKots
  name/id lookup, lazy CMTM/CMVector-compatible views, and state_dict conversion.
  """

  def __init__(self, robot: RobotStruct, raw_data, order: int):
    self.robot = robot
    self.raw_data = raw_data
    self.order = int(order)
    self.link_ids = {link.name: i for i, link in enumerate(robot.links)}
    self.joint_ids = {joint.name: i for i, joint in enumerate(robot.joints)}
    self.joint_dofs = tuple(joint.dof for joint in robot.joints)
    self._cache = {}
    self._has_kinematics = False
    self._has_dynamics = False
    self._minimal_dynamics = False

  def compute_kinematics(self, motion) -> "RustOutwardState":
    motion = np.asarray(motion, dtype=float)
    if motion.ndim != 1:
      raise ValueError("motion must have shape (robot dof * order,)")
    self.raw_data.compute_kinematics(motion)
    self._cache.clear()
    self._has_kinematics = True
    self._has_dynamics = False
    self._minimal_dynamics = False
    return self

  def compute_dynamics(self, motion) -> "RustOutwardState":
    motion = np.asarray(motion, dtype=float)
    if motion.ndim != 1:
      raise ValueError("motion must have shape (robot dof * order,)")
    self.raw_data.compute_dynamics(motion)
    self._cache.clear()
    self._has_kinematics = True
    self._has_dynamics = True
    self._minimal_dynamics = False
    return self

  def compute_dynamics_minimal(self, motion) -> "RustOutwardState":
    motion = np.asarray(motion, dtype=float)
    if motion.ndim != 1:
      raise ValueError("motion must have shape (robot dof * order,)")
    self.raw_data.compute_dynamics_minimal(motion)
    self._cache.clear()
    self._has_kinematics = True
    self._has_dynamics = True
    self._minimal_dynamics = True
    return self

  def _require_full_dynamics(self):
    if self._minimal_dynamics:
      raise ValueError("compute_dynamics_minimal only stores kinematics and joint torque; use compute_dynamics for full dynamics values")

  def to_state_dict(self, robot: RobotStruct | None = None) -> dict:
    if robot is None:
      robot = self.robot
    state_dict = {}

    for link in robot.links:
      state_dict.update(cmtm_to_state_list(self.cmtm("link", link.name, self.order), "link", link.name))

    for joint in robot.joints:
      state_dict.update(cmtm_to_state_list(self.cmtm("joint", joint.name, self.order), "joint", joint.name))

    momentum_order = self.order - 1
    if self._has_dynamics and momentum_order > 0:
      for link in robot.links:
        momentum = self.cmvec("link", link.name, "momentum")
        state_dict.update(vecs_to_state_dict(momentum.vecs(), "link", link.name, "momentum", momentum_order))
      for joint in robot.joints:
        momentum = self.cmvec("joint", joint.name, "momentum")
        state_dict.update(vecs_to_state_dict(momentum.vecs(), "joint", joint.name, "momentum", momentum_order))

    force_order = self.order - 2
    if self._has_dynamics and force_order > 0:
      for link in robot.links:
        force = self.cmvec("link", link.name, "force")
        state_dict.update(vecs_to_state_dict(force.vecs(), "link", link.name, "force", force_order))
      for joint in robot.joints:
        force = self.cmvec("joint", joint.name, "force")
        state_dict.update(vecs_to_state_dict(force.vecs(), "joint", joint.name, "force", force_order))

      for joint in robot.joints:
        dof = self.joint_dofs[self._joint_id(joint.name)]
        if dof <= 0:
          continue
        torque = np.stack(
          [self.joint_torque(joint.name, key_order) for key_order in range(1, force_order + 1)],
          axis=-2,
        )
        state_dict.update(vecs_to_state_dict(torque, "joint", joint.name, "torque", force_order))

    return state_dict

  def link_mat(self, link) -> np.ndarray:
    return np.asarray(self.raw_data.link_mat(self._link_id(link)))

  def joint_mat(self, joint) -> np.ndarray:
    return np.asarray(self.raw_data.joint_mat(self._joint_id(joint)))

  def link_vec(self, link, key_order: int) -> np.ndarray:
    return np.asarray(self.raw_data.link_vec(self._link_id(link), int(key_order)))

  def joint_vec(self, joint, key_order: int) -> np.ndarray:
    return np.asarray(self.raw_data.joint_vec(self._joint_id(joint), int(key_order)))

  def link_momentum(self, link, key_order: int) -> np.ndarray:
    self._require_full_dynamics()
    return np.asarray(self.raw_data.link_momentum(self._link_id(link), int(key_order)))

  def joint_momentum(self, joint, key_order: int) -> np.ndarray:
    self._require_full_dynamics()
    return np.asarray(self.raw_data.joint_momentum(self._joint_id(joint), int(key_order)))

  def world_link_momentum(self, link, key_order: int) -> np.ndarray:
    self._require_full_dynamics()
    return np.asarray(self.raw_data.world_link_momentum(self._link_id(link), int(key_order)))

  def world_joint_momentum(self, joint, key_order: int) -> np.ndarray:
    self._require_full_dynamics()
    return np.asarray(self.raw_data.world_joint_momentum(self._joint_id(joint), int(key_order)))

  def link_force(self, link, key_order: int) -> np.ndarray:
    self._require_full_dynamics()
    return np.asarray(self.raw_data.link_force(self._link_id(link), int(key_order)))

  def joint_force(self, joint, key_order: int) -> np.ndarray:
    self._require_full_dynamics()
    return np.asarray(self.raw_data.joint_force(self._joint_id(joint), int(key_order)))

  def world_link_force(self, link, key_order: int) -> np.ndarray:
    self._require_full_dynamics()
    return np.asarray(self.raw_data.world_link_force(self._link_id(link), int(key_order)))

  def world_joint_force(self, joint, key_order: int) -> np.ndarray:
    self._require_full_dynamics()
    return np.asarray(self.raw_data.world_joint_force(self._joint_id(joint), int(key_order)))

  def joint_torque(self, joint, key_order: int) -> np.ndarray:
    joint_id = self._joint_id(joint)
    dof = self.joint_dofs[joint_id]
    if dof == 0:
      return np.zeros((0,), dtype=float)
    return np.asarray(self.raw_data.joint_torque(joint_id, int(key_order)))[..., :dof]

  def cmtm(self, owner_type: str, owner_name: str, order: int | None = None) -> _RustCMTMView:
    if order is None:
      order = self.order
    order = int(order)
    cache_key = ("cmtm", owner_type, owner_name, order)
    if cache_key not in self._cache:
      mat = self._mat(owner_type, owner_name)
      vecs = self._vecs(owner_type, owner_name, order)
      self._cache[cache_key] = _RustCMTMView(
        mat,
        vecs,
        rust_wrench_kernel=self.raw_data.cmtm_wrench_var_jacob_matvec,
      )
    return self._cache[cache_key]

  def cmtm_wrench(self, owner_type: str, owner_name: str, order: int | None = None) -> _RustCMTMView:
    if order is None:
      order = self.order
    order = int(order)
    cache_key = ("cmtm_wrench", owner_type, owner_name, order)
    if cache_key not in self._cache:
      self._cache[cache_key] = self.cmtm(owner_type, owner_name, order).as_wrench()
    return self._cache[cache_key]

  def rel_cmtm(
    self,
    base_name: str,
    target_name: str,
    owner_type: str = "link",
    order: int | None = None,
  ) -> _RustCMTMView:
    if order is None:
      order = self.order
    order = int(order)
    cache_key = ("rel_cmtm", owner_type, base_name, target_name, order)
    if cache_key not in self._cache:
      self._cache[cache_key] = (
        self.cmtm(owner_type, base_name, order).inv()
        @ self.cmtm(owner_type, target_name, order)
      )
    return self._cache[cache_key]

  def rel_cmtm_wrench(
    self,
    base_name: str,
    target_name: str,
    owner_type: str = "link",
    order: int | None = None,
  ) -> _RustCMTMView:
    if order is None:
      order = self.order
    order = int(order)
    cache_key = ("rel_cmtm_wrench", owner_type, base_name, target_name, order)
    if cache_key not in self._cache:
      self._cache[cache_key] = self.rel_cmtm(base_name, target_name, owner_type, order).as_wrench()
    return self._cache[cache_key]

  def cmvec(self, owner_type: str, owner_name: str, data_type: str) -> _RustCMVectorView:
    cache_key = ("cmvec", owner_type, owner_name, data_type)
    if cache_key not in self._cache:
      if data_type == "momentum":
        order = self.order - 1
        getter = self._momentum
      elif data_type == "force":
        order = self.order - 2
        getter = self._force
      else:
        raise KeyError((owner_type, owner_name, data_type))
      if order < 1:
        raise ValueError(f"{data_type} is not available for order={self.order}")
      self._cache[cache_key] = _RustCMVectorView(
        np.stack(
          [getter(owner_type, owner_name, key_order) for key_order in range(1, order + 1)],
          axis=-2,
        )
      )
    return self._cache[cache_key]

  def state_value(self, state_type):
    if state_type.data_type == "cmtm":
      return self.cmtm(state_type.owner_type, state_type.owner_name, state_type.time_order)
    if state_type.owner_type not in ("link", "joint"):
      raise NotImplementedError(
        f"RustOutwardState direct state access does not support owner_type={state_type.owner_type!r}"
      )

    name = state_type.owner_name
    owner_type = state_type.owner_type
    data_type = state_type.data_type
    key_order = int(state_type.key_order)

    if data_type == "frame":
      return SE3.set_mat(self._mat(owner_type, name))
    if data_type == "pos":
      return self._mat(owner_type, name)[..., :3, 3]
    if data_type == "rot":
      mat = self._mat(owner_type, name)
      return mat[..., :3, :3].reshape(mat.shape[:-2] + (9,))
    if data_type in keys_kinematics:
      if key_order < 2:
        raise NotImplementedError(f"Unsupported kinematics data_type={data_type!r}")
      return self._vec(owner_type, name, key_order)
    if data_type in keys_momentum:
      if state_type.frame_name == "world":
        return self._world_momentum(owner_type, name, key_order)
      return self._momentum(owner_type, name, key_order)
    if data_type in keys_force:
      if state_type.frame_name == "world":
        return self._world_force(owner_type, name, key_order)
      return self._force(owner_type, name, key_order)
    if data_type in keys_torque:
      if owner_type != "joint":
        raise NotImplementedError("Torque values are only defined for joint owners")
      return self.joint_torque(name, key_order)
    raise NotImplementedError(f"Unsupported data_type={data_type!r}")

  def _mat(self, owner_type: str, name) -> np.ndarray:
    if owner_type == "link":
      return self.link_mat(name)
    return self.joint_mat(name)

  def _vec(self, owner_type: str, name, key_order: int) -> np.ndarray:
    if owner_type == "link":
      return self.link_vec(name, key_order)
    return self.joint_vec(name, key_order)

  def _vecs(self, owner_type: str, name, order: int) -> np.ndarray:
    if order < 1 or order > self.order:
      raise ValueError(f"Invalid order: {order}. Must be in 1..={self.order}.")
    if order == 1:
      return np.zeros(self._mat(owner_type, name).shape[:-2] + (0, 6), dtype=float)
    return np.stack(
      [self._vec(owner_type, name, key_order) for key_order in range(2, order + 1)],
      axis=-2,
    )

  def _momentum(self, owner_type: str, name, key_order: int) -> np.ndarray:
    if owner_type == "link":
      return self.link_momentum(name, key_order)
    return self.joint_momentum(name, key_order)

  def _world_momentum(self, owner_type: str, name, key_order: int) -> np.ndarray:
    if owner_type == "link":
      return self.world_link_momentum(name, key_order)
    return self.world_joint_momentum(name, key_order)

  def _force(self, owner_type: str, name, key_order: int) -> np.ndarray:
    if owner_type == "link":
      return self.link_force(name, key_order)
    return self.joint_force(name, key_order)

  def _world_force(self, owner_type: str, name, key_order: int) -> np.ndarray:
    if owner_type == "link":
      return self.world_link_force(name, key_order)
    return self.world_joint_force(name, key_order)

  def _link_id(self, link) -> int:
    if isinstance(link, str):
      return self.link_ids[link]
    return int(link)

  def _joint_id(self, joint) -> int:
    if isinstance(joint, str):
      return self.joint_ids[joint]
    return int(joint)


def create_rust_outward_state(
  robot: RobotStruct,
  order: int,
  compiled_robot=None,
) -> RustOutwardState:
  if order < 1:
    raise ValueError("order must be >= 1")
  rust_robot = compiled_robot if compiled_robot is not None else _rust_compiled_robot(robot)
  return RustOutwardState(robot, rust_robot.create_outward_data(order), order)


class RustBatchOutwardState(RustOutwardState):
  """Batched Rust outward-state adapter.

  The raw Rust workspace stores flattened samples; this subclass restores the
  original batch shape at the Python outward-state boundary.
  """

  def __init__(self, robot: RobotStruct, raw_data, order: int, batch_shape: tuple[int, ...]):
    super().__init__(robot, raw_data, order)
    self.batch_shape = tuple(batch_shape)
    self.batch_size = int(np.prod(self.batch_shape, dtype=int)) if self.batch_shape else 1

  def compute_kinematics(self, motion) -> "RustBatchOutwardState":
    flat_motion = self._flat_motion(motion)
    self.raw_data.compute_kinematics(flat_motion)
    self._cache.clear()
    self._has_kinematics = True
    self._has_dynamics = False
    self._minimal_dynamics = False
    return self

  def compute_dynamics(self, motion) -> "RustBatchOutwardState":
    flat_motion = self._flat_motion(motion)
    self.raw_data.compute_dynamics(flat_motion)
    self._cache.clear()
    self._has_kinematics = True
    self._has_dynamics = True
    self._minimal_dynamics = False
    return self

  def compute_dynamics_minimal(self, motion) -> "RustBatchOutwardState":
    flat_motion = self._flat_motion(motion)
    self.raw_data.compute_dynamics_minimal(flat_motion)
    self._cache.clear()
    self._has_kinematics = True
    self._has_dynamics = True
    self._minimal_dynamics = True
    return self

  def link_mat(self, link) -> np.ndarray:
    return self._reshape_mat(self.raw_data.link_mat(self._link_id(link)))

  def joint_mat(self, joint) -> np.ndarray:
    return self._reshape_mat(self.raw_data.joint_mat(self._joint_id(joint)))

  def link_vec(self, link, key_order: int) -> np.ndarray:
    return self._reshape_vec(self.raw_data.link_vec(self._link_id(link), int(key_order)))

  def joint_vec(self, joint, key_order: int) -> np.ndarray:
    return self._reshape_vec(self.raw_data.joint_vec(self._joint_id(joint), int(key_order)))

  def link_momentum(self, link, key_order: int) -> np.ndarray:
    self._require_full_dynamics()
    return self._reshape_vec(self.raw_data.link_momentum(self._link_id(link), int(key_order)))

  def joint_momentum(self, joint, key_order: int) -> np.ndarray:
    self._require_full_dynamics()
    return self._reshape_vec(self.raw_data.joint_momentum(self._joint_id(joint), int(key_order)))

  def world_link_momentum(self, link, key_order: int) -> np.ndarray:
    self._require_full_dynamics()
    return self._reshape_vec(self.raw_data.world_link_momentum(self._link_id(link), int(key_order)))

  def world_joint_momentum(self, joint, key_order: int) -> np.ndarray:
    self._require_full_dynamics()
    return self._reshape_vec(self.raw_data.world_joint_momentum(self._joint_id(joint), int(key_order)))

  def link_force(self, link, key_order: int) -> np.ndarray:
    self._require_full_dynamics()
    return self._reshape_vec(self.raw_data.link_force(self._link_id(link), int(key_order)))

  def joint_force(self, joint, key_order: int) -> np.ndarray:
    self._require_full_dynamics()
    return self._reshape_vec(self.raw_data.joint_force(self._joint_id(joint), int(key_order)))

  def world_link_force(self, link, key_order: int) -> np.ndarray:
    self._require_full_dynamics()
    return self._reshape_vec(self.raw_data.world_link_force(self._link_id(link), int(key_order)))

  def world_joint_force(self, joint, key_order: int) -> np.ndarray:
    self._require_full_dynamics()
    return self._reshape_vec(self.raw_data.world_joint_force(self._joint_id(joint), int(key_order)))

  def joint_torque(self, joint, key_order: int) -> np.ndarray:
    joint_id = self._joint_id(joint)
    dof = self.joint_dofs[joint_id]
    raw = np.asarray(self.raw_data.joint_torque(joint_id, int(key_order)))[..., :dof]
    return raw.reshape(self.batch_shape + (dof,))

  def _flat_motion(self, motion) -> np.ndarray:
    motion = np.asarray(motion, dtype=float)
    flat_motion, batch_shape = batch_api.flatten_feature_batch(motion)
    if tuple(batch_shape) != self.batch_shape:
      raise ValueError(f"motion batch shape must be {self.batch_shape}, got {batch_shape}")
    return flat_motion

  def _reshape_mat(self, value) -> np.ndarray:
    return np.asarray(value).reshape(self.batch_shape + (4, 4))

  def _reshape_vec(self, value) -> np.ndarray:
    return np.asarray(value).reshape(self.batch_shape + (6,))


def create_rust_batch_outward_state(
  robot: RobotStruct,
  order: int,
  batch_shape: tuple[int, ...],
  compiled_robot=None,
) -> RustBatchOutwardState:
  if order < 1:
    raise ValueError("order must be >= 1")
  batch_shape = tuple(batch_shape)
  if not batch_shape:
    raise ValueError("batch_shape must be non-empty")
  batch_size = int(np.prod(batch_shape, dtype=int))
  rust_robot = compiled_robot if compiled_robot is not None else _rust_compiled_robot(robot)
  return RustBatchOutwardState(
    robot,
    rust_robot.create_batch_outward_data(order, batch_size),
    order,
    batch_shape,
  )


# Backward-compatible aliases for in-tree benchmark scripts while the Rust
# backend remains experimental. Prefer the *state names in new code.
create_rust_outward_data = create_rust_outward_state
create_rust_batch_outward_data = create_rust_batch_outward_state
