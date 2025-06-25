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
"""Pure Python implementation of the Trimmed Match estimator."""

import dataclasses
import math
from typing import List, Set, NamedTuple

import numpy as np
from scipy import stats

from trimmed_match.geox_data_util import GeoxDataUtil, GeoPairValues
from trimmed_match.math_util import trimmed_symmetric_norm, square

class TrimAndError(NamedTuple):
  trim_rate: float
  iroas: float
  std_error: float

@dataclasses.dataclass
class Report:
  """A class to report the Trimmed Match estimator."""
  estimate: float
  std_error: float
  trim_rate: float
  confidence: float
  conf_interval_low: float
  conf_interval_up: float
  trimmed_pairs_indices: Set[int]
  candidate_results: List[TrimAndError]

class TrimmedMatch(object):
  """The TrimmedMatch estimator."""

  def __init__(self,
               delta_response: List[float],
               delta_spend: List[float],
               max_trim_rate: float = 0.25):
    """Initializes the class."""

    def CheckForTies():
      """Checks if delta_spend or {theta[i,j]: i<j} has duplicated values."""
      dresponse = np.array(delta_response)
      dspend = np.array(delta_spend)

      # check ties in delta_spend
      dspend_has_ties = (len(np.unique(dspend)) != len(dspend))

      if not dspend_has_ties:
        # check ties in thetaij
        delta2_response = dresponse[:, None] - dresponse[None, :]
        delta2_spend = dspend[:, None] - dspend[None, :]
        upper_indices = np.triu_indices(len(dresponse), k=1)
        thetaij = delta2_response[upper_indices] / delta2_spend[upper_indices]
        thetaij_has_ties = (len(np.unique(thetaij)) != len(thetaij))

        if not thetaij_has_ties:
          return 0
        else:
          return 1
      else:
        return 2

    if len(delta_response) != len(delta_spend):
      raise ValueError("Lengths of delta_response and delta_spend differ.")

    if max_trim_rate < 0.0:
      raise ValueError("max_trim_rate is negative.")

    if np.max(np.abs(delta_spend)) < 1e-10:
      raise ValueError("delta_spends are all too close to 0!")

    self._max_trim_rate = max_trim_rate
    self._num_pairs = len(delta_response)
    
    ties = CheckForTies()
    if ties == 0:
      perturb_dspend = perturb_dresponse = np.zeros(len(delta_response))
      perturbation = 0
    else:
      perturbation = 2 ** (-40)
      perturb_dresponse = np.arange(len(delta_response))**1.5
      perturb_dresponse = perturb_dresponse - np.median(perturb_dresponse)
      if ties == 2:
        perturb_dspend = np.arange(len(delta_spend)) - len(delta_spend) * 0.5
      else:
        perturb_dspend = np.zeros(len(delta_response))
    perturb_dspend, perturb_dresponse = [
        perturb_dspend * perturbation,
        perturb_dresponse * perturbation
    ]

    _VectorPerturb = lambda x, y: (x + np.finfo(float).tiny) * (1 + y)
    
    dresponse = _VectorPerturb(np.array(delta_response), perturb_dresponse)
    dspend = _VectorPerturb(np.array(delta_spend), perturb_dspend)
    
    geox_data = [GeoPairValues(r, c) for r, c in zip(dresponse, dspend)]
    self._geox_util = GeoxDataUtil(geox_data)

  def calculate_iroas(self, trim_rate: float) -> float:
    """Calculates the iROAS for a given trim rate."""
    if not (0.0 <= trim_rate <= self._max_trim_rate):
      raise ValueError(f"Trim rate must be in [0, {self._max_trim_rate}]")

    if trim_rate == 0.0:
      return self._geox_util.calculate_empirical_iroas()

    candidates = self._geox_util.find_all_zeros_of_trimmed_mean(trim_rate)
    if not candidates:
      raise ValueError("Incremental cost for the untrimmed geo pairs is 0")

    if len(candidates) == 1:
      return candidates[0]

    # If multiple candidates exist, find the one that minimizes TrimmedSymmetricNorm.
    min_norm = float('inf')
    best_iroas = 0.0
    for iroas in candidates:
      residuals = self._geox_util.calculate_residuals(iroas)
      norm = trimmed_symmetric_norm(residuals.tolist(), trim_rate)
      if norm < min_norm:
        min_norm = norm
        best_iroas = iroas
        
    return best_iroas

  def calculate_standard_error(self, trim_rate: float, iroas: float) -> float:
    """Calculates the standard error of the iROAS estimate."""
    if not (0.0 <= trim_rate <= self._max_trim_rate):
      raise ValueError(f"Trim rate must be in [0, {self._max_trim_rate}]")

    residuals = self._geox_util.calculate_residuals(iroas)
    delta_cost = self._geox_util.extract_delta_cost()
    
    n1 = math.ceil(trim_rate * self._num_pairs)
    n2 = self._num_pairs - 1 - n1

    # Sort residuals and delta_cost together
    sorted_indices = np.argsort(residuals)
    sorted_residuals = residuals[sorted_indices]
    sorted_delta_cost = delta_cost[sorted_indices]

    # Calculate winsorized squared sum of residuals
    winsorized_sum_sq_res = (
        n1 * (square(sorted_residuals[n1]) + square(sorted_residuals[n2]))
    )
    winsorized_sum_sq_res += np.sum(square(sorted_residuals[n1 : n2 + 1]))

    # Calculate trimmed mean of delta_cost
    trimmed_mean_delta_cost = np.sum(sorted_delta_cost[n1 : n2 + 1]) / self._num_pairs
    
    if abs(trimmed_mean_delta_cost) < 1e-9:
        return float('inf')

    approx_variance = winsorized_sum_sq_res / (
        square(trimmed_mean_delta_cost) * self._num_pairs
    )

    return math.sqrt(approx_variance / self._num_pairs)

  def estimate(self, trim_rate: float):
    """Compute iROAS and standard error for a specified trim rate."""
    iroas = self.calculate_iroas(trim_rate)
    std_error = self.calculate_standard_error(trim_rate, iroas)
    return iroas, std_error

  def auto_estimate(self):
    """Automatically select trim rate via one-SE rule.

    Returns:
        (estimate, std_error, selected_trim_rate)
    """
    max_num_trim = math.ceil(self._max_trim_rate * self._num_pairs)
    candidate_results = []
    for i in range(max_num_trim + 1):
      rate = i / self._num_pairs
      if rate > self._max_trim_rate:
        break
      iroas = self.calculate_iroas(rate)
      std_error = self.calculate_standard_error(rate, iroas)
      candidate_results.append((rate, iroas, std_error))

    # Select using one-SE rule (same logic as in report())
    min_error = min(c[2] for c in candidate_results)
    threshold = (1.0 + 0.25 / math.sqrt(self._num_pairs)) * min_error
    for rate, iroas, std_error in candidate_results:
      if std_error <= threshold:
        return iroas, std_error, rate
    # Fallback: minimal error
    rate, iroas, std_error = min(candidate_results, key=lambda x: x[2])
    return iroas, std_error, rate

  def confidence_interval(self, trim_rate: float, confidence: float = 0.9):
    """Return (low, up) confidence interval for a trim rate."""
    normal_quantile = stats.norm.ppf(0.5 + 0.5 * confidence)
    low, up = self._geox_util.range_from_studentized_trimmed_mean(trim_rate, normal_quantile)
    if up - low > 1e6:
      iroas, std_error = self.estimate(trim_rate)
      width = std_error * normal_quantile * 3.0
      low = iroas - width
      up = iroas + width
    return low, up

  def report(self, confidence: float = 0.80, trim_rate: float = -1.0) -> Report:
    """Reports the Trimmed Match estimation."""
    if not (0.0 < confidence <= 1.0):
      raise ValueError("Confidence must be in (0, 1]")
    if trim_rate > self._max_trim_rate:
      raise ValueError(f"trim_rate {trim_rate} > max_trim_rate {self._max_trim_rate}")

    candidate_results = []
    if 0.0 <= trim_rate <= self._max_trim_rate:
      iroas = self.calculate_iroas(trim_rate)
      std_error = self.calculate_standard_error(trim_rate, iroas)
      candidate_results.append(TrimAndError(trim_rate, iroas, std_error))
      result = candidate_results[0]
    else:
      max_num_trim = math.ceil(self._max_trim_rate * self._num_pairs)
      for i in range(max_num_trim + 1):
        rate = i / self._num_pairs
        if rate > self._max_trim_rate:
          break
        iroas = self.calculate_iroas(rate)
        std_error = self.calculate_standard_error(rate, iroas)
        candidate_results.append(TrimAndError(rate, iroas, std_error))
      
      min_error = min(c.std_error for c in candidate_results)
      one_se_rule_threshold = (1.0 + 0.25 / math.sqrt(self._num_pairs)) * min_error
      
      result = next(c for c in candidate_results if c.std_error <= one_se_rule_threshold)

    # Confidence interval
    normal_quantile = stats.norm.ppf(0.5 + 0.5 * confidence)
    conf_interval_low, conf_interval_up = self._geox_util.range_from_studentized_trimmed_mean(
        result.trim_rate, normal_quantile
    )
    
    if conf_interval_up - conf_interval_low > 1e6 * result.std_error:
        width = result.std_error * normal_quantile * 3.0
        conf_interval_low = result.iroas - width
        conf_interval_up = result.iroas + width

    # Trimmed pairs
    residuals = self._geox_util.calculate_residuals(result.iroas)
    ranks = np.argsort(np.argsort(residuals))
    num_pairs = len(ranks)
    left_trim = round(num_pairs * result.trim_rate)
    trimmed_pairs_indices = {
        i for i, rank in enumerate(ranks) 
        if rank < left_trim or rank > num_pairs - left_trim - 1
    }

    return Report(
        estimate=result.iroas,
        std_error=result.std_error,
        trim_rate=result.trim_rate,
        confidence=confidence,
        conf_interval_low=conf_interval_low,
        conf_interval_up=conf_interval_up,
        trimmed_pairs_indices=trimmed_pairs_indices,
        candidate_results=candidate_results,
    )
