from __future__ import annotations

import os
from math import factorial

import numpy as np


def _dense_apply(mat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
  mat = np.asarray(mat)
  rhs = np.asarray(rhs)
  if rhs.ndim >= 2 and rhs.shape[-2] == mat.shape[-1]:
    return mat @ rhs
  if rhs.shape[-1] != mat.shape[-1]:
    raise ValueError("rhs has incompatible trailing dimension")
  if rhs.ndim == 1:
    return mat @ rhs
  return (mat @ rhs[..., None])[..., 0]


def _has_batch(cmtm) -> bool:
  return np.asarray(cmtm.elem_mat()).ndim > 2


def _use_direct_apply(cmtm) -> bool:
  return os.environ.get("ROBOKOTS_CMTM_APPLY", "1") != "0" and _has_batch(cmtm)


def apply_mat_adj(cmtm, rhs: np.ndarray) -> np.ndarray:
  apply = getattr(cmtm, "apply_mat_adj", None)
  if apply is not None and _use_direct_apply(cmtm):
    return apply(rhs)
  return _dense_apply(cmtm.mat_adj(), rhs)


def apply_mat_inv_adj(cmtm, rhs: np.ndarray) -> np.ndarray:
  apply = getattr(cmtm, "apply_mat_inv_adj", None)
  if apply is not None and _use_direct_apply(cmtm):
    return apply(rhs)
  return _dense_apply(cmtm.mat_inv_adj(), rhs)


def apply_tangent_mat(cmtm, rhs: np.ndarray) -> np.ndarray:
  apply = getattr(cmtm, "apply_tangent_mat", None)
  if apply is not None and _use_direct_apply(cmtm):
    return apply(rhs)
  return _dense_apply(cmtm.tangent_mat(), rhs)


def last_ordinary_derivative(coefficients: np.ndarray) -> np.ndarray:
  """Restore only the last row of a normalized derivative series.

  Match CMVector.vecs()'s NumPy dtype promotion without reconstructing earlier
  rows. The returned value is independent of the coefficient storage.
  """
  coefficients = np.asarray(coefficients)
  order = coefficients.shape[-2]
  if order < 1:
    raise ValueError("At least one coefficient row is required")
  row = coefficients[..., -1, :]
  dtype = np.result_type(row.dtype, np.float64)
  return row.astype(dtype, copy=False) * factorial(order - 1)


def world_cmvector_value(vector, frame, order: int) -> np.ndarray:
  """Transform CM coefficients and restore only the requested derivative."""
  coefficients = np.asarray(vector.cm_vecs())
  if order < 1 or order > coefficients.shape[-2]:
    raise ValueError("Requested order is outside the coefficient series")
  coefficients = coefficients[..., :order, :]
  flat = coefficients.reshape(coefficients.shape[:-2] + (-1,))
  transformed = apply_mat_adj(frame, flat[..., None])[..., 0]
  rows = transformed.reshape(transformed.shape[:-1] + (order, coefficients.shape[-1]))
  return last_ordinary_derivative(rows)


def world_spatial_value(owner, frame, order: int) -> np.ndarray:
  """Last ordinary derivative of Ad(frame) applied to a local motion series."""
  return world_cmvector_value(owner.cmvecs(), frame, order)
