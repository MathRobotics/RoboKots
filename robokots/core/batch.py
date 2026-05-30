from __future__ import annotations

from typing import Callable, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def is_batched_feature_array(data, feature_ndim: int = 1) -> bool:
  return np.asarray(data).ndim > feature_ndim


def flatten_feature_batch(data, feature_ndim: int = 1) -> tuple[np.ndarray, tuple[int, ...]]:
  arr = np.asarray(data, dtype=float)
  if feature_ndim < 1:
    raise ValueError("feature_ndim must be greater than 0")
  if arr.ndim < feature_ndim:
    raise ValueError(f"data must have at least {feature_ndim} dimensions, got {arr.ndim}")
  feature_shape = arr.shape[-feature_ndim:]
  if arr.ndim == feature_ndim:
    return arr.reshape((1,) + feature_shape), ()
  batch_shape = arr.shape[:-feature_ndim]
  return arr.reshape((-1,) + feature_shape), batch_shape


def map_flat_batch(data, fn: Callable[[np.ndarray], T], feature_ndim: int = 1) -> tuple[T | list[T], tuple[int, ...]]:
  flat_data, batch_shape = flatten_feature_batch(data, feature_ndim=feature_ndim)
  values = [fn(x) for x in flat_data]
  if not batch_shape:
    return values[0], ()
  return values, batch_shape


def stack_batch_values(values: Sequence, batch_shape: tuple[int, ...]):
  first = values[0]
  if hasattr(first, "mat"):
    stacked = np.stack([np.asarray(v.mat()) for v in values], axis=0)
  else:
    try:
      stacked = np.stack([np.asarray(v) for v in values], axis=0)
    except Exception:
      return np.asarray(values, dtype=object).reshape(batch_shape)
  return stacked.reshape(batch_shape + stacked.shape[1:])


def stack_sample_results(values: Sequence, batch_shape: tuple[int, ...]):
  stacked = np.stack([np.asarray(v) for v in values], axis=0)
  return stacked.reshape(batch_shape + stacked.shape[1:])


def stack_list_results(values: Sequence[Sequence], batch_shape: tuple[int, ...], item_count: int):
  return [
    stack_sample_results([sample[i] for sample in values], batch_shape)
    for i in range(item_count)
  ]


def stack_results(values: Sequence, batch_shape: tuple[int, ...], list_output: bool, item_count: int, combine = None):
  if list_output:
    return stack_list_results(values, batch_shape, item_count)
  if combine is not None:
    values = [combine(value) for value in values]
  return stack_sample_results(values, batch_shape)


def broadcast_feature_vector(vec, batch_shape: tuple[int, ...], feature_shape: tuple[int, ...], name: str = "vec"):
  arr = np.asarray(vec)
  if not batch_shape:
    if arr.shape != feature_shape:
      raise ValueError(f"{name} must have shape {feature_shape}, got {arr.shape}")
    return arr

  batch_expected_shape = batch_shape + feature_shape
  if arr.shape == feature_shape:
    return np.broadcast_to(arr, batch_expected_shape).reshape((-1,) + feature_shape)
  if arr.shape == batch_expected_shape:
    return arr.reshape((-1,) + feature_shape)
  raise ValueError(f"{name} must have shape {feature_shape} or {batch_expected_shape}, got {arr.shape}")
