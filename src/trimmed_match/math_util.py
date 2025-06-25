# Copyright 2025 Charles Shaw (charles@fixedpoint.io).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Pure Python implementation of math utilities for Trimmed Match."""

import math
from typing import List
import numpy as np

def square(x: float) -> float:
  """Returns the square of a number."""
  return x * x

def trimmed_mean(x: List[float], trim_rate: float) -> float:
  """Calculates the trimmed mean of a list of numbers."""
  if not (0 <= trim_rate < 0.5):
    raise ValueError("Trim rate must be in [0, 0.5)")

  n = len(x)
  if n == 0:
    return np.nan

  num_trims = math.ceil(n * trim_rate)
  if 2 * num_trims >= n:
    return np.nan

  sorted_x = np.sort(np.array(x))
  return np.mean(sorted_x[num_trims : n - num_trims])

def studentized_trimmed_mean(x: List[float], trim_rate: float) -> float:
  """Calculates the studentized trimmed mean."""
  if not (0 <= trim_rate < 0.5):
    raise ValueError("Trim rate must be in [0, 0.5)")

  n = len(x)
  if n == 0:
    return np.nan

  num_trims = math.ceil(n * trim_rate)
  if 2 * num_trims >= n:
    return np.nan

  sorted_x = np.sort(np.array(x))
  trimmed_x = sorted_x[num_trims : n - num_trims]
  mean_trimmed = np.mean(trimmed_x)

  # Winsorized sum of squares
  lower_bound = sorted_x[num_trims]
  upper_bound = sorted_x[n - 1 - num_trims]
  winsorized_x = np.clip(sorted_x, lower_bound, upper_bound)

  h = n - 2 * num_trims
  winsorized_sum_sq = np.sum(square(winsorized_x - mean_trimmed))

  if h <= 1:
    return np.inf if mean_trimmed != 0 else 0

  winsorized_var = winsorized_sum_sq / (h - 1)

  if winsorized_var == 0:
    return np.inf if mean_trimmed != 0 else 0

  std_err_trimmed_mean = math.sqrt(winsorized_var * n) / h

  return mean_trimmed / std_err_trimmed_mean if std_err_trimmed_mean > 1e-9 else np.inf

def trimmed_symmetric_norm(x: List[float], trim_rate: float) -> float:
  """Calculates the trimmed symmetric norm of a list of numbers.

  Args:
    x: A list of numbers.
    trim_rate: The trim rate.

  Returns:
    The trimmed symmetric norm.
  """
  if not (0 <= trim_rate < 0.5):
    raise ValueError("Trim rate must be in [0, 0.5)")

  n = len(x)
  if n == 0:
    return 0.0

  sorted_x = np.sort(np.array(x))
  num_trims = math.ceil(n * trim_rate)

  if 2 * num_trims >= n:
    return 0.0

  untrimmed_x = sorted_x[num_trims : n - num_trims]

  # The norm measures deviation from symmetry. For a perfectly symmetric
  # distribution around 0, we'd have x_i = -x_{n-1-i}.
  # This norm is based on the sum of |x_i + x_{n-1-i}|.
  reversed_untrimmed_x = untrimmed_x[::-1]
  # The sum below correctly calculates Sum(|x_i + x_{n-1-i}|) for the untrimmed
  # part of the data, handling both even and odd number of elements correctly.
  # We divide by 2 to correct for double counting.
  return np.sum(np.abs(untrimmed_x + reversed_untrimmed_x)) / 2.0
