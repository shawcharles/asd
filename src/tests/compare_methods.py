"""Compare the three methods: Trimmed Match, Supergeos, and Adaptive Supergeos.

This script runs all three methods on synthetic datasets and compares their performance.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import time

from trimmed_match.estimator import TrimmedMatch
from supergeos.supergeo_design import GeoUnit, SupergeoDesign
from adaptive_supergeos.adaptive_supergeo_design import AdaptiveSupergeoDesign


def load_dataset(file_path: str) -> Dict[str, Any]:
    """Load a dataset from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary with the dataset
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Add the dataset name
    data["name"] = os.path.splitext(os.path.basename(file_path))[0]
    
    return data


def run_trimmed_match(
    dataset: Dict[str, Any],
    confidence: float = 0.9,
    trim_rate: Optional[float] = None
) -> Dict[str, Any]:
    """Run the Trimmed Match algorithm on a dataset.
    
    Args:
        dataset: Dictionary with the dataset
        confidence: Confidence level for the interval
        trim_rate: Optional trim rate (if None, will be auto-selected)
        
    Returns:
        Dictionary with results
    """
    delta_response = dataset["delta_response"]
    delta_spend = dataset["delta_spend"]
    
    # Create and run the TrimmedMatch estimator
    estimator = TrimmedMatch(delta_response, delta_spend)
    
    # Run with specified trim rate or auto-select
    if trim_rate is not None:
        estimate, std_error = estimator.estimate(trim_rate)
        conf_interval_low, conf_interval_up = estimator.confidence_interval(
            trim_rate, confidence)
    else:
        estimate, std_error, selected_trim_rate = estimator.auto_estimate()
        conf_interval_low, conf_interval_up = estimator.confidence_interval(
            selected_trim_rate, confidence)
        trim_rate = selected_trim_rate
    
    # Return the results
    return {
        "method": "TrimmedMatch",
        "estimate": estimate,
        "std_error": std_error,
        "conf_interval_low": conf_interval_low,
        "conf_interval_up": conf_interval_up,
        "trim_rate": trim_rate
    }


def run_supergeo_design(
    dataset: Dict[str, Any],
    n_treatment_supergeos: int = 5,
    n_control_supergeos: int = 5,
    min_supergeo_size: int = 2,
    max_supergeo_size: int = 10,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """Run the Supergeo Design algorithm on a dataset.
    
    Args:
        dataset: Dictionary with the dataset
        n_treatment_supergeos: Number of treatment supergeos to create
        n_control_supergeos: Number of control supergeos to create
        min_supergeo_size: Minimum number of units in a supergeo
        max_supergeo_size: Maximum number of units in a supergeo
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with results
    """
    # Convert the dataset to GeoUnit objects
    geo_units = []
    for i in range(len(dataset["delta_response"])):
        geo_units.append(
            GeoUnit(
                id=str(i),
                response=dataset["delta_response"][i],
                spend=dataset["delta_spend"][i]
            )
        )
    
    # Create and run the SupergeoDesign
    supergeo_design = SupergeoDesign(
        geo_units=geo_units,
        min_supergeo_size=min_supergeo_size,
        max_supergeo_size=max_supergeo_size
    )
    
    start_time = time.time()
    
    partition = supergeo_design.optimize_partition(
        n_treatment_supergeos=n_treatment_supergeos,
        n_control_supergeos=n_control_supergeos,
        random_seed=random_seed
    )
    
    end_time = time.time()
    
    # Evaluate the partition
    evaluation = supergeo_design.evaluate_partition(partition)
    
    # Return the results
    return {
        "method": "SupergeoDesign",
        "balance_score": partition.balance_score,
        "covariate_balance": partition.covariate_balance,
        "treatment_units": sum(len(sg) for sg in partition.treatment_supergeos),
        "control_units": sum(len(sg) for sg in partition.control_supergeos),
        "runtime_seconds": end_time - start_time
    }


def run_adaptive_supergeo_design(
    dataset: Dict[str, Any],
    n_treatment_supergeos: int = 5,
    n_control_supergeos: int = 5,
    min_supergeo_size: int = 2,
    max_supergeo_size: int = 10,
    embedding_dim: int = 32,
    batch_size: int = 100,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """Run the Adaptive Supergeo Design algorithm on a dataset.
    
    Args:
        dataset: Dictionary with the dataset
        n_treatment_supergeos: Number of treatment supergeos to create
        n_control_supergeos: Number of control supergeos to create
        min_supergeo_size: Minimum number of units in a supergeo
        max_supergeo_size: Maximum number of units in a supergeo
        embedding_dim: Dimension of the embeddings
        batch_size: Number of candidate supergeos to generate in each batch
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with results
    """
    # Convert the dataset to GeoUnit objects
    geo_units = []
    for i in range(len(dataset["delta_response"])):
        geo_units.append(
            GeoUnit(
                id=str(i),
                response=dataset["delta_response"][i],
                spend=dataset["delta_spend"][i]
            )
        )
    
    # Create and run the AdaptiveSupergeoDesign
    adaptive_supergeo_design = AdaptiveSupergeoDesign(
        geo_units=geo_units,
        min_supergeo_size=min_supergeo_size,
        max_supergeo_size=max_supergeo_size,
        embedding_dim=embedding_dim,
        batch_size=batch_size
    )
    
    start_time = time.time()
    
    partition = adaptive_supergeo_design.optimize_partition(
        n_treatment_supergeos=n_treatment_supergeos,
        n_control_supergeos=n_control_supergeos,
        random_seed=random_seed
    )
    
    end_time = time.time()
    
    # Evaluate the partition
    evaluation = adaptive_supergeo_design.evaluate_partition(partition)
    
    # Return the results
    return {
        "method": "AdaptiveSupergeoDesign",
        "balance_score": partition.balance_score,
        "covariate_balance": partition.covariate_balance,
        "treatment_units": sum(len(sg) for sg in partition.treatment_supergeos),
        "control_units": sum(len(sg) for sg in partition.control_supergeos),
        "runtime_seconds": end_time - start_time
    }


def run_comparisons_on_all_datasets(
    data_dir: str,
    confidence: float = 0.9,
    n_treatment_supergeos: int = 5,
    n_control_supergeos: int = 5,
    random_seed: Optional[int] = None,
    output_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Run comparison tests on all datasets in a directory.
    
    Args:
        data_dir: Directory containing the datasets
        confidence: Confidence level for the interval
        n_treatment_supergeos: Number of treatment supergeos to create
        n_control_supergeos: Number of control supergeos to create
        random_seed: Random seed for reproducibility
        output_file: Optional file to save the results to
        
    Returns:
        List of dictionaries with comparison results
    """
    # Get all dataset files
    dataset_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                    if f.endswith(".json")]
    
    all_results = []
    
    # Run comparisons for each dataset
    for dataset_file in dataset_files:
        dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Load the dataset
        dataset = load_dataset(dataset_file)
        
        # Run Trimmed Match with auto trim rate
        trimmed_match_auto = run_trimmed_match(dataset, confidence, None)
        trimmed_match_auto["dataset_name"] = dataset_name
        trimmed_match_auto["trim_rate_type"] = "auto"
        all_results.append(trimmed_match_auto)
        
        # Run Trimmed Match with fixed trim rates
        for trim_rate in [0.0, 0.1, 0.2]:
            trimmed_match_fixed = run_trimmed_match(dataset, confidence, trim_rate)
            trimmed_match_fixed["dataset_name"] = dataset_name
            trimmed_match_fixed["trim_rate_type"] = f"fixed_{trim_rate}"
            all_results.append(trimmed_match_fixed)
        
        # Run Supergeo Design
        supergeo_design = run_supergeo_design(
            dataset,
            n_treatment_supergeos=n_treatment_supergeos,
            n_control_supergeos=n_control_supergeos,
            random_seed=random_seed
        )
        supergeo_design["dataset_name"] = dataset_name
        all_results.append(supergeo_design)
        
        # Run Adaptive Supergeo Design
        adaptive_supergeo_design = run_adaptive_supergeo_design(
            dataset,
            n_treatment_supergeos=n_treatment_supergeos,
            n_control_supergeos=n_control_supergeos,
            random_seed=random_seed
        )
        adaptive_supergeo_design["dataset_name"] = dataset_name
        all_results.append(adaptive_supergeo_design)
    
    # Save results to file if specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
    
    return all_results


def print_results_table(results: List[Dict[str, Any]]):
    """Print a table of results.
    
    Args:
        results: List of dictionaries with results
    """
    # Group results by dataset and method
    datasets = sorted(set(r["dataset_name"] for r in results))
    
    # Print header
    print("\nComparison Results:")
    header = f"{'Dataset':<15} {'Method':<20} {'Estimate':<10} {'95% CI':<30} {'Runtime (s)':<12}"
    print(header)
    print("-" * len(header))
    
    # Print results for each dataset
    for dataset in datasets:
        dataset_results = [r for r in results if r["dataset_name"] == dataset]
        
        for result in dataset_results:
            method = result.get("method", "Unknown")
            
            if method == "TrimmedMatch":
                trim_type = result.get("trim_rate_type", "")
                method_str = f"{method} ({trim_type})"
                estimate = f"{result.get('estimate', 'N/A'):.4f}"
                ci = f"[{result.get('conf_interval_low', 'N/A'):.4f}, {result.get('conf_interval_up', 'N/A'):.4f}]"
                runtime = "N/A"
            else:
                method_str = method
                estimate = "N/A"
                ci = "N/A"
                runtime = f"{result.get('runtime_seconds', 'N/A'):.4f}"
            
            print(f"{dataset:<15} {method_str:<20} {estimate:<10} {ci:<30} {runtime:<12}")


def main():
    """Run the comparison script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare Trimmed Match, Supergeos, and Adaptive Supergeos"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../tests/data",
        help="Directory containing the datasets"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.9,
        help="Confidence level for the interval"
    )
    parser.add_argument(
        "--n-treatment",
        type=int,
        default=5,
        help="Number of treatment supergeos to create"
    )
    parser.add_argument(
        "--n-control",
        type=int,
        default=5,
        help="Number of control supergeos to create"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional file to save the results to"
    )
    
    args = parser.parse_args()
    
    results = run_comparisons_on_all_datasets(
        data_dir=args.data_dir,
        confidence=args.confidence,
        n_treatment_supergeos=args.n_treatment,
        n_control_supergeos=args.n_control,
        random_seed=args.random_seed,
        output_file=args.output_file
    )
    
    print_results_table(results)


if __name__ == "__main__":
    main()
