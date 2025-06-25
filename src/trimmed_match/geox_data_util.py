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
"""Pure Python implementation of GeoxDataUtil for Trimmed Match."""

import math
from typing import List, NamedTuple, Tuple
import numpy as np
from trimmed_match.quadratic_inequality import QuadraticInequality
from trimmed_match.math_util import square, trimmed_mean

class PairedDelta(NamedTuple):
  i: int
  j: int
  delta: float

class GeoPairValues(NamedTuple):
  """Represents the values for a pair of geos."""
  delta_response: float
  delta_cost: float

class GeoxDataUtil:
  """A utility class for handling geo experiment data."""

  def __init__(self, geox_data: List[GeoPairValues]):
    """Initializes the GeoxDataUtil.

    Args:
      geox_data: A list of GeoPairValues.
    """
    self._geox_data = sorted(geox_data, key=lambda p: (p.delta_cost, p.delta_response))
    self._num_pairs = len(self._geox_data)
    self._delta_response = np.array([p.delta_response for p in self._geox_data])
    self._delta_cost = np.array([p.delta_cost for p in self._geox_data])
    self._paired_delta_sorted = self._get_paired_delta_sorted()

  def calculate_residuals(self, iroas: float) -> np.ndarray:
    """Calculates the residuals for a given iROAS."""
    return self._delta_response - iroas * self._delta_cost

  def extract_delta_cost(self) -> np.ndarray:
    """Returns the delta cost array."""
    return self._delta_cost

  def calculate_empirical_iroas(self) -> float:
    """Calculates the empirical iROAS."""
    sum_delta_cost = np.sum(self._delta_cost)
    if np.abs(sum_delta_cost) < 1e-9:
      raise ValueError("Sum of delta_cost is zero, cannot compute empirical iROAS.")
    return np.sum(self._delta_response) / sum_delta_cost

  def _get_paired_delta_sorted(self) -> List[PairedDelta]:
    """Calculates and sorts the pairwise deltas."""
    paired_deltas = []
    for i in range(1, self._num_pairs):
        for j in range(i):
            delta_resp = self._delta_response[i] - self._delta_response[j]
            delta_cost = self._delta_cost[i] - self._delta_cost[j]
            if abs(delta_cost) > 1e-9:
                paired_deltas.append(PairedDelta(i, j, delta_resp / delta_cost))
    
    return sorted(paired_deltas, key=lambda p: (p.delta, p.i, p.j))

  def find_all_zeros_of_trimmed_mean(self, trim_rate: float) -> List[float]:
    """Finds all values of iroas that are roots of the trimmed mean equation."""
    if self._num_pairs < 2:
        return []

    def get_trimmed_mean_at_iroas(iroas):
        residuals = self.calculate_residuals(iroas)
        return trimmed_mean(residuals.tolist(), trim_rate)

    if not self._paired_delta_sorted:
        return []

    # Check the sign of trimmed mean at a point far to the left of all candidates
    start_delta = self._paired_delta_sorted[0].delta - 1.0
    prev_tm = get_trimmed_mean_at_iroas(start_delta)

    if np.isnan(prev_tm):
        return []

    unique_zeros = set()
    prev_delta = start_delta

    for p_delta in self._paired_delta_sorted:
        delta = p_delta.delta
        # If deltas are too close, skip to avoid redundant calculations
        if abs(delta - prev_delta) < 1e-9:
            continue

        current_tm = get_trimmed_mean_at_iroas(delta)
        if np.isnan(current_tm):
            continue

        if abs(current_tm) < 1e-9:
            unique_zeros.add(delta)
        elif prev_tm * current_tm < 0:
            # Root is between prev_delta and delta. The set of trimmed indices is
            # constant in this interval. Find the root by solving the linear equation.
            residuals_at_prev = self.calculate_residuals(prev_delta)
            sorted_indices = np.argsort(residuals_at_prev)

            num_trims = math.ceil(self._num_pairs * trim_rate)
            untrimmed_indices = sorted_indices[num_trims : self._num_pairs - num_trims]

            sum_resp = np.sum(self._delta_response[untrimmed_indices])
            sum_cost = np.sum(self._delta_cost[untrimmed_indices])

            if abs(sum_cost) > 1e-9:
                zero = sum_resp / sum_cost
                unique_zeros.add(zero)

        prev_tm = current_tm
        prev_delta = delta

    return sorted(list(unique_zeros))

  def range_from_studentized_trimmed_mean(
      self, trim_rate: float, normal_quantile: float
  ) -> Tuple[float, float]:
    """Calculates the confidence interval for the iROAS estimate."""
    if not (trim_rate >= 0.0 and normal_quantile > 0.0 and self._num_pairs >= 2):
        raise ValueError("Invalid arguments for confidence interval calculation.")

    if trim_rate == 0.0:
        h = (square(normal_quantile) * self._num_pairs) / (self._num_pairs - 1)
        x1 = np.sum(self._delta_cost)
        y1 = np.sum(self._delta_response)
        x2 = np.sum(square(self._delta_cost))
        y2 = np.sum(square(self._delta_response))
        z = np.sum(self._delta_response * self._delta_cost)
        r = 1.0 + h / self._num_pairs
        # We solve a*x^2 - 2*b*x + c >= 0, so we pass the negated coefficients
        # to the Python solver which solves for <= 0.
        qi = QuadraticInequality(-(h * x2 - r * square(x1)), -(h * z - r * x1 * y1), -(h * y2 - r * square(y1)))
        return qi.solver(float('-inf'), float('inf'))

    num_trims = math.ceil(trim_rate * self._num_pairs)
    if self._num_pairs < 2 * num_trims + 2:
        raise ValueError("Less than 2 values are left after trimming")

    h = self._num_pairs - 2 * num_trims
    threshold_h_sqrt = normal_quantile * math.sqrt(h * (h - 1))

    min_max_delta = self._delta_range(num_trims, self._num_pairs - 1 - num_trims)
    conf_interval = (min_max_delta.delta_min, min_max_delta.delta_max)

    for p_delta in self._paired_delta_sorted:
        delta = p_delta.delta
        if delta < conf_interval[0] or delta > conf_interval[1]:
            continue

        residuals = self.calculate_residuals(delta)
        sorted_indices = np.argsort(residuals)
        
        untrimmed_indices = sorted_indices[num_trims : self._num_pairs - num_trims]
        winsorized_indices = np.concatenate([
            np.repeat(sorted_indices[num_trims], num_trims),
            untrimmed_indices,
            np.repeat(sorted_indices[self._num_pairs - 1 - num_trims], num_trims)
        ])

        x = self._delta_cost[untrimmed_indices]
        y = self._delta_response[untrimmed_indices]
        x_w = self._delta_cost[winsorized_indices]
        y_w = self._delta_response[winsorized_indices]

        A = np.sum(square(x))
        B = np.sum(x * y)
        C = np.sum(square(y))

        a_w = np.sum(square(x_w))
        b_w = np.sum(x_w * y_w)
        c_w = np.sum(square(y_w))

        a = h * (h - 1) * A - square(threshold_h_sqrt) * a_w
        b = h * (h - 1) * B - square(threshold_h_sqrt) * b_w
        c = h * (h - 1) * C - square(threshold_h_sqrt) * c_w

        qi = QuadraticInequality(-a, -b, -c)
        current_interval = qi.solver(conf_interval[0], conf_interval[1])

        if not np.isnan(current_interval[0]):
            conf_interval = (max(conf_interval[0], current_interval[0]), min(conf_interval[1], current_interval[1]))

    return conf_interval

  def _delta_range(self, n1: int, n2: int) -> NamedTuple:
    """Calculates the min and max delta values."""
    min_val = float('inf')
    max_val = float('-inf')
    for i in range(n1, n2 + 1):
        min_max = self._delta_relative_to_one_geo_pair(i)
        min_val = min(min_val, min_max.delta_min)
        max_val = max(max_val, min_max.delta_max)
    return min_max

  def _delta_relative_to_one_geo_pair(self, index: int) -> NamedTuple:
    """Calculates the min and max delta relative to one geo pair."""
    min_val = float('inf')
    max_val = float('-inf')
    for i in range(self._num_pairs):
        if i == index:
            continue
        delta_cost = self._delta_cost[i] - self._delta_cost[index]
        if abs(delta_cost) > 1e-9:
            delta = (self._delta_response[i] - self._delta_response[index]) / delta_cost
            min_val = min(min_val, delta)
            max_val = max(max_val, delta)
    DeltaMinMax = NamedTuple('DeltaMinMax', [('delta_min', float), ('delta_max', float)])
    return DeltaMinMax(min_val, max_val)
