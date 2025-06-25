import numpy as np
from supergeos.supergeo_design import GeoUnit
from adaptive_supergeos.adaptive_supergeo_design import AdaptiveSupergeoDesign
from supergeos.scoring import absolute_difference


def _make_synthetic_units(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    units = []
    for i in range(n):
        response = rng.normal(loc=100.0, scale=10.0)
        spend = rng.normal(loc=50.0, scale=5.0)
        units.append(GeoUnit(id=str(i), response=response, spend=spend))
    return units


def test_greedy_assignment_improves_balance():
    """The greedy ASD partition should improve balance vs a naive split."""
    geo_units = _make_synthetic_units(30)

    # Naive split: first half treatment, second half control (single supergeo each)
    naive_t = geo_units[:15]
    naive_c = geo_units[15:]
    naive_score = absolute_difference(naive_t, naive_c)

    # Adaptive Supergeo Design with greedy assignment
    asd = AdaptiveSupergeoDesign(geo_units, min_supergeo_size=2, max_supergeo_size=6, batch_size=50)
    partition = asd.optimize_partition(
        n_treatment_supergeos=3,
        n_control_supergeos=3,
        random_seed=1,
        max_iterations=100,
    )

    # Basic sanity checks on partition counts
    assert len(partition.treatment_supergeos) == 3, "Unexpected number of treatment supergeos"
    assert len(partition.control_supergeos) == 3, "Unexpected number of control supergeos"

    # The greedy balance score should be at least as good (lower) than naive
    assert partition.balance_score <= naive_score, (
        f"Greedy balance score {partition.balance_score} should not exceed naive {naive_score}")


def test_deterministic_with_fixed_seed():
    geo_units = _make_synthetic_units(20)
    asd1 = AdaptiveSupergeoDesign(geo_units, batch_size=40)
    part1 = asd1.optimize_partition(2, 2, random_seed=123, max_iterations=50)

    asd2 = AdaptiveSupergeoDesign(geo_units, batch_size=40)
    part2 = asd2.optimize_partition(2, 2, random_seed=123, max_iterations=50)

    assert abs(part1.balance_score - part2.balance_score) < 1e-9, "Results should be deterministic with fixed seed"
