"""Supergeo Design implementation for geographic experiment design.

This module implements the original Supergeo Design algorithm as described in
the literature, which groups geographic units to create better matches for
experimental design.
"""

import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .scoring import absolute_difference, relative_difference, covariate_diffs


@dataclass
class GeoUnit:
    """Represents a geographic unit with its properties."""
    id: str
    response: float
    spend: float
    covariates: Optional[Dict[str, float]] = None


@dataclass
class SupergeoPartition:
    """Represents a partition of geographic units into supergeos."""
    treatment_supergeos: List[List[GeoUnit]]
    control_supergeos: List[List[GeoUnit]]
    balance_score: float
    covariate_balance: Dict[str, float]


class SupergeoDesign:
    """Implementation of the original Supergeo Design algorithm.
    
    This class implements the original Supergeo Design algorithm which
    formulates the partitioning problem as an NP-hard optimization problem
    to find optimal groupings of geographic units.
    """
    
    def __init__(
        self,
        geo_units: List[GeoUnit],
        min_supergeo_size: int = 2,
        max_supergeo_size: int = 10,
        balance_metric: str = "absolute_difference"
    ):
        """Initialize the SupergeoDesign.
        
        Args:
            geo_units: List of geographic units to partition
            min_supergeo_size: Minimum number of units in a supergeo
            max_supergeo_size: Maximum number of units in a supergeo
            balance_metric: Metric to use for measuring balance ("absolute_difference" or "relative_difference")
        """
        self.geo_units = geo_units
        self.min_supergeo_size = min_supergeo_size
        self.max_supergeo_size = max_supergeo_size
        self.balance_metric = balance_metric

    def _balance_score(self, treatment: List[GeoUnit], control: List[GeoUnit]) -> float:
        """Compute the overall balance score according to the selected metric."""
        if self.balance_metric == "absolute_difference":
            return absolute_difference(treatment, control)
        return relative_difference(treatment, control)

        
    def optimize_partition(
        self,
        n_treatment_supergeos: int,
        n_control_supergeos: int,
        balance_weights: Optional[Dict[str, float]] = None,
        max_iterations: int = 1000,
        random_seed: Optional[int] = None
    ) -> SupergeoPartition:
        """Find an optimal partition of geographic units into supergeos.
        
        Args:
            n_treatment_supergeos: Number of treatment supergeos to create
            n_control_supergeos: Number of control supergeos to create
            balance_weights: Optional weights for balancing different metrics
            max_iterations: Maximum number of iterations for optimization
            random_seed: Random seed for reproducibility
            
        Returns:
            A SupergeoPartition object with the optimal partition
        """
        # Greedy ratio-matching implementation
        if random_seed is not None:
            np.random.seed(random_seed)

        # Compute ratio; fall back to response if spend≈0
        def ratio(g: GeoUnit):
            return g.response / g.spend if abs(g.spend) > 1e-9 else g.response

        units_sorted = sorted(self.geo_units, key=ratio)

        # Greedily pair adjacent units (nearest ratio)
        pairs: List[List[GeoUnit]] = []
        i = 0
        while i < len(units_sorted) - 1:
            pairs.append([units_sorted[i], units_sorted[i + 1]])
            i += 2
        if i < len(units_sorted):  # odd leftover
            pairs[-1].append(units_sorted[i])

        # Combine pairs into supergeos up to desired counts
        treatment_supergeos: List[List[GeoUnit]] = []
        control_supergeos: List[List[GeoUnit]] = []
        toggle = True
        for pair in pairs:
            if len(treatment_supergeos) < n_treatment_supergeos and toggle:
                treatment_supergeos.append(pair)
            elif len(control_supergeos) < n_control_supergeos:
                control_supergeos.append(pair)
            else:
                # Overflow — append to whichever set is smaller
                if len(treatment_supergeos) < n_treatment_supergeos:
                    treatment_supergeos[-1].extend(pair)
                else:
                    control_supergeos[-1].extend(pair)
            toggle = not toggle

        # -------------------------------------------------------------------
        # Local search refinement – swap individual geo units between the two
        # arms if it reduces the overall balance score. We only perform swaps
        # that keep supergeos within the size bounds.
        # -------------------------------------------------------------------
        flat_treatment = [g for sg in treatment_supergeos for g in sg]
        flat_control = [g for sg in control_supergeos for g in sg]
        current_score = self._balance_score(flat_treatment, flat_control)

        iteration = 0
        improved = True
        while improved and iteration < max_iterations:
            improved = False
            best_new_score = current_score
            best_swap: Optional[Tuple[int, int, int, int]] = None  # (ti, tj, ci, cj)

            # Enumerate possible swaps
            for ti, t_sg in enumerate(treatment_supergeos):
                for tj, g_t in enumerate(t_sg):
                    for ci, c_sg in enumerate(control_supergeos):
                        for cj, g_c in enumerate(c_sg):
                            # Swapping does not change sizes, so bounds hold.
                            new_flat_t = flat_treatment.copy()
                            new_flat_c = flat_control.copy()
                            # Perform swap in temporary lists
                            new_flat_t.remove(g_t)
                            new_flat_c.remove(g_c)
                            new_flat_t.append(g_c)
                            new_flat_c.append(g_t)
                            new_score = self._balance_score(new_flat_t, new_flat_c)
                            if new_score < best_new_score - 1e-12:  # strict improvement
                                best_new_score = new_score
                                best_swap = (ti, tj, ci, cj)

            # Apply the best swap if it improves the score
            if best_swap is not None:
                ti, tj, ci, cj = best_swap
                g_t = treatment_supergeos[ti][tj]
                g_c = control_supergeos[ci][cj]
                treatment_supergeos[ti][tj] = g_c
                control_supergeos[ci][cj] = g_t

                # Update flattened lists & score
                flat_treatment.remove(g_t)
                flat_control.remove(g_c)
                flat_treatment.append(g_c)
                flat_control.append(g_t)
                current_score = best_new_score
                improved = True
                iteration += 1
            else:
                break

        # Final balance calculations after refinement
        balance_score = current_score

        covariate_balance = covariate_diffs(
            [g for sg in treatment_supergeos for g in sg],
            [g for sg in control_supergeos for g in sg],
        )
        
        return SupergeoPartition(
            treatment_supergeos=treatment_supergeos,
            control_supergeos=control_supergeos,
            balance_score=balance_score,
            covariate_balance=covariate_balance
        )
    
    def evaluate_partition(self, partition: SupergeoPartition) -> Dict[str, float]:
        """Evaluate the quality of a partition.
        
        Args:
            partition: A SupergeoPartition to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        return {
            "balance_score": partition.balance_score,
            "covariate_balance": partition.covariate_balance,
            "treatment_units": sum(len(sg) for sg in partition.treatment_supergeos),
            "control_units": sum(len(sg) for sg in partition.control_supergeos),
        }
