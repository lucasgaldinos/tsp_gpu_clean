"""Deterministic ``uint64`` SplitMix64 PRNG matching ``utils/kernels/splitmix64.cuh`` (`Task 17.31`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from numpy.random import Generator

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_MUL2 = np.uint64(0x94D049BB133111EB)
_SHIFT30 = np.uint64(30)
_SHIFT27 = np.uint64(27)
_SHIFT31 = np.uint64(31)
_SHIFT32 = np.uint64(32)


def draw_row_seeds(rng: Generator, population_size: int) -> np.ndarray:
  """Draw ``population_size`` independent ``uint64`` row seeds from a NumPy ``Generator``."""
  if population_size <= 0:
    return np.empty(0, dtype=np.uint64)
  return rng.integers(0, np.iinfo(np.int64).max, size=population_size, dtype=np.int64).view(np.uint64)


def splitmix64_step(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Advance a ``uint64`` state array by one SplitMix64 step, returning ``(new_states, z)``."""
  with np.errstate(over="ignore"):
    new_states = states + _GAMMA
    z = (new_states ^ (new_states >> _SHIFT30)) * _MUL1
    z = (z ^ (z >> _SHIFT27)) * _MUL2
    z = z ^ (z >> _SHIFT31)
  return new_states, z


def splitmix64_next_scalar(state: int) -> tuple[int, int]:
  """Advance one scalar ``uint64`` state by one SplitMix64 step, returning ``(new_state, z)``."""
  mask = 0xFFFFFFFFFFFFFFFF
  new_state = (state + 0x9E3779B97F4A7C15) & mask
  z = ((new_state ^ (new_state >> 30)) * 0xBF58476D1CE4E5B9) & mask
  z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
  z = (z ^ (z >> 31)) & mask
  return new_state, z


def splitmix64_bounded_scalar(state: int, bound: int) -> tuple[int, int]:
  """Advance ``state`` and return ``(new_state, value)`` with ``value`` in ``[0, bound)``."""
  new_state, z = splitmix64_next_scalar(state)
  return new_state, ((z >> 32) * bound) >> 32


def splitmix64_bounded(z: np.ndarray, bound: int | np.ndarray) -> np.ndarray:
  """Map ``uint64`` values ``z`` into ``[0, bound)`` via high-32 multiply."""
  return ((z >> _SHIFT32) * np.asarray(bound, dtype=np.uint64)) >> _SHIFT32


__all__: list[str] = [
  "draw_row_seeds",
  "splitmix64_bounded",
  "splitmix64_bounded_scalar",
  "splitmix64_next_scalar",
  "splitmix64_step",
]
