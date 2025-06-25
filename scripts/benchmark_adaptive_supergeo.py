import pathlib
import sys
import time

# Ensure we can import project packages when script is run directly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
import numpy as np
from supergeos.supergeo_design import GeoUnit
from adaptive_supergeos.adaptive_supergeo_design import AdaptiveSupergeoDesign


def make_dataset(n_units: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    units = []
    for i in range(n_units):
        response = rng.normal(100.0, 15.0)
        spend = rng.normal(50.0, 8.0)
        units.append(GeoUnit(id=str(i), response=response, spend=spend))
    return units


def benchmark(n_units: int = 200, n_reps: int = 5):
    units = make_dataset(n_units)
    runtimes = []
    scores = []
    for rep in range(n_reps):
        start = time.perf_counter()
        asd = AdaptiveSupergeoDesign(units, min_supergeo_size=2, max_supergeo_size=8, batch_size=100)
        part = asd.optimize_partition(4, 4, max_iterations=200, random_seed=rep)
        runtimes.append(time.perf_counter() - start)
        scores.append(part.balance_score)
    print("--- Adaptive Supergeo Design Benchmark ---")
    print(f"Units: {n_units} | Repetitions: {n_reps}")
    print(f"Mean runtime: {np.mean(runtimes):.3f} s (±{np.std(runtimes):.3f})")
    print(f"Mean balance score: {np.mean(scores):.4f} (±{np.std(scores):.4f})")


if __name__ == "__main__":
    benchmark()
