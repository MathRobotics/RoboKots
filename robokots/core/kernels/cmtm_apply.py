from __future__ import annotations

import os

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
