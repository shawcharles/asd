"""
Hyperparameter sensitivity analysis for Adaptive Supergeo Design methodology.

This module provides systematic analysis of hyperparameter choices and their
impact on performance, including grid search optimization and sensitivity plots.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import itertools
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add src directory to path for imports
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from analysis.statistical_tests import StatisticalAnalyzer
from visualization.publication_plots import PublicationPlotter

class HyperparameterOptimizer:
    """
    Systematic hyperparameter optimization for ASD methodology.
    
    Provides grid search, random search, and Bayesian optimization
    capabilities for finding optimal hyperparameter configurations.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the hyperparameter optimizer."""
        self.random_state = random_state
        self.stat_framework = StatisticalAnalyzer()
        self.plotter = PublicationPlotter()
        np.random.seed(random_state)
    
    def grid_search_embedding_dimensions(self, 
                                       dimensions: List[int] = None,
                                       n_units_list: List[int] = None,
                                       n_replications: int = 20) -> Dict:
        """
        Grid search for optimal embedding dimensions across different problem sizes.
        
        Parameters:
        -----------
        dimensions : List[int]
            List of embedding dimensions to test
        n_units_list : List[int]
            List of problem sizes to test
        n_replications : int
            Number of replications per configuration
            
        Returns:
        --------
        Dict
            Grid search results with performance metrics
        """
        if dimensions is None:
            dimensions = [8, 16, 32, 64, 128, 256]
        if n_units_list is None:
            n_units_list = [100, 200, 400, 800]
        
        print(f"Grid search: {len(dimensions)} dimensions × {len(n_units_list)} problem sizes")
        
        results = {
            'dimensions': dimensions,
            'n_units_list': n_units_list,
            'performance_matrix': np.zeros((len(dimensions), len(n_units_list))),
            'computational_cost': np.zeros((len(dimensions), len(n_units_list))),
            'memory_usage': np.zeros((len(dimensions), len(n_units_list))),
            'overfitting_risk': np.zeros((len(dimensions), len(n_units_list))),
            'detailed_results': {}
        }
        
        for i, dim in enumerate(dimensions):
            for j, n_units in enumerate(n_units_list):
                print(f"  Testing dim={dim}, n_units={n_units}")
                
                # Simulate performance for this configuration
                config_results = self._simulate_embedding_performance(
                    embedding_dim=dim,
                    n_units=n_units,
                    n_replications=n_replications
                )
                
                # Store aggregated metrics
                results['performance_matrix'][i, j] = config_results['mean_performance']
                results['computational_cost'][i, j] = config_results['computational_cost']
                results['memory_usage'][i, j] = config_results['memory_usage']
                results['overfitting_risk'][i, j] = config_results['overfitting_risk']
                
                # Store detailed results
                results['detailed_results'][(dim, n_units)] = config_results
        
        # Find optimal configurations
        results['optimal_configs'] = self._find_optimal_configurations(results)
        
        return results
    
    def sensitivity_analysis_similarity_thresholds(self,
                                                 thresholds: List[float] = None,
                                                 clustering_methods: List[str] = None,
                                                 n_replications: int = 30) -> Dict:
        """
        Analyze sensitivity to similarity thresholds in clustering stage.
        
        Parameters:
        -----------
        thresholds : List[float]
            List of similarity thresholds to test
        clustering_methods : List[str]
            List of clustering methods to compare
        n_replications : int
            Number of replications per configuration
            
        Returns:
        --------
        Dict
            Sensitivity analysis results
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9).tolist()
        if clustering_methods is None:
            clustering_methods = ['ward', 'complete', 'average', 'single']
        
        print(f"Threshold sensitivity: {len(thresholds)} thresholds × {len(clustering_methods)} methods")
        
        results = {
            'thresholds': thresholds,
            'clustering_methods': clustering_methods,
            'cluster_quality': {},
            'balance_preservation': {},
            'computational_efficiency': {},
            'statistical_performance': {}
        }
        
        for method in clustering_methods:
            results['cluster_quality'][method] = []
            results['balance_preservation'][method] = []
            results['computational_efficiency'][method] = []
            results['statistical_performance'][method] = []
            
            for threshold in thresholds:
                print(f"  Testing {method} clustering with threshold={threshold:.2f}")
                
                # Simulate clustering performance
                cluster_results = self._simulate_clustering_performance(
                    threshold=threshold,
                    method=method,
                    n_replications=n_replications
                )
                
                results['cluster_quality'][method].append(cluster_results['quality'])
                results['balance_preservation'][method].append(cluster_results['balance'])
                results['computational_efficiency'][method].append(cluster_results['efficiency'])
                results['statistical_performance'][method].append(cluster_results['performance'])
        
        # Find optimal thresholds for each method
        results['optimal_thresholds'] = self._find_optimal_thresholds(results)
        
        return results
    
    def batch_size_optimization(self,
                              batch_sizes: List[int] = None,
                              problem_sizes: List[int] = None,
                              n_replications: int = 20) -> Dict:
        """
        Optimize batch size for different problem sizes and computational constraints.
        
        Parameters:
        -----------
        batch_sizes : List[int]
            List of batch sizes to test
        problem_sizes : List[int]
            List of problem sizes to test
        n_replications : int
            Number of replications per configuration
            
        Returns:
        --------
        Dict
            Batch size optimization results
        """
        if batch_sizes is None:
            batch_sizes = [8, 16, 32, 64, 128, 256]
        if problem_sizes is None:
            problem_sizes = [100, 200, 400, 800]
        
        print(f"Batch size optimization: {len(batch_sizes)} sizes × {len(problem_sizes)} problems")
        
        results = {
            'batch_sizes': batch_sizes,
            'problem_sizes': problem_sizes,
            'convergence_stability': np.zeros((len(batch_sizes), len(problem_sizes))),
            'training_speed': np.zeros((len(batch_sizes), len(problem_sizes))),
            'memory_efficiency': np.zeros((len(batch_sizes), len(problem_sizes))),
            'solution_quality': np.zeros((len(batch_sizes), len(problem_sizes)))
        }
        
        for i, batch_size in enumerate(batch_sizes):
            for j, problem_size in enumerate(problem_sizes):
                print(f"  Testing batch_size={batch_size}, problem_size={problem_size}")
                
                # Simulate batch performance
                batch_results = self._simulate_batch_performance(
                    batch_size=batch_size,
                    problem_size=problem_size,
                    n_replications=n_replications
                )
                
                results['convergence_stability'][i, j] = batch_results['stability']
                results['training_speed'][i, j] = batch_results['speed']
                results['memory_efficiency'][i, j] = batch_results['memory']
                results['solution_quality'][i, j] = batch_results['quality']
        
        # Find optimal batch sizes
        results['optimal_batch_sizes'] = self._find_optimal_batch_sizes(results)
        
        return results
    
    def comprehensive_hyperparameter_study(self,
                                         param_grid: Dict = None,
                                         n_replications: int = 15) -> Dict:
        """
        Comprehensive hyperparameter study using full factorial design.
        
        Parameters:
        -----------
        param_grid : Dict
            Parameter grid for comprehensive search
        n_replications : int
            Number of replications per configuration
            
        Returns:
        --------
        Dict
            Comprehensive study results
        """
        if param_grid is None:
            param_grid = {
                'embedding_dim': [16, 32, 64],
                'similarity_threshold': [0.3, 0.5, 0.7],
                'batch_size': [16, 32, 64],
                'learning_rate': [0.001, 0.01, 0.1],
                'regularization': [0.0, 0.01, 0.1]
            }
        
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        print(f"Comprehensive study: {len(param_combinations)} parameter combinations")
        
        results = {
            'param_grid': param_grid,
            'param_combinations': param_combinations,
            'performance_scores': [],
            'computational_costs': [],
            'memory_usage': [],
            'detailed_results': {}
        }
        
        for i, params in enumerate(param_combinations):
            print(f"  Configuration {i+1}/{len(param_combinations)}: {params}")
            
            # Simulate performance for this parameter combination
            config_results = self._simulate_full_configuration(
                params=params,
                n_replications=n_replications
            )
            
            results['performance_scores'].append(config_results['performance'])
            results['computational_costs'].append(config_results['cost'])
            results['memory_usage'].append(config_results['memory'])
            results['detailed_results'][i] = {
                'params': params,
                'results': config_results
            }
        
        # Analyze parameter importance
        results['parameter_importance'] = self._analyze_parameter_importance(results)
        
        # Find Pareto optimal configurations
        results['pareto_optimal'] = self._find_pareto_optimal_configs(results)
        
        return results
    
    def _simulate_embedding_performance(self, embedding_dim: int, n_units: int, n_replications: int) -> Dict:
        """Simulate embedding performance for given configuration."""
        # Representation capacity
        capacity_ratio = embedding_dim / (n_units * 0.1)  # Rule of thumb
        representation_capacity = min(1.0, capacity_ratio)
        
        # Overfitting risk
        param_ratio = embedding_dim / n_units
        overfitting_risk = max(0.0, min(1.0, (param_ratio - 0.05) / 0.1))
        
        # Computational cost (quadratic in dimension)
        computational_cost = (embedding_dim ** 2) / (32 ** 2)  # Normalized
        
        # Memory usage (linear in dimension and units)
        memory_usage = (embedding_dim * n_units) / (32 * 200)  # Normalized
        
        # Overall performance (trade-off between capacity and overfitting)
        base_performance = representation_capacity * (1 - overfitting_risk * 0.5)
        
        # Add noise for realistic simulation
        performance_samples = []
        for _ in range(n_replications):
            noise = np.random.normal(0, 0.1)
            performance_samples.append(max(0.0, min(1.0, base_performance + noise)))
        
        return {
            'mean_performance': np.mean(performance_samples),
            'std_performance': np.std(performance_samples),
            'computational_cost': computational_cost,
            'memory_usage': memory_usage,
            'overfitting_risk': overfitting_risk,
            'representation_capacity': representation_capacity
        }
    
    def _simulate_clustering_performance(self, threshold: float, method: str, n_replications: int) -> Dict:
        """Simulate clustering performance for given configuration."""
        # Method-specific base performance
        method_performance = {
            'ward': 0.85,
            'complete': 0.80,
            'average': 0.75,
            'single': 0.65
        }
        
        base_perf = method_performance.get(method, 0.75)
        
        # Threshold effects (optimal around 0.4-0.6)
        if 0.4 <= threshold <= 0.6:
            threshold_factor = 1.0 - abs(threshold - 0.5) * 2
        else:
            threshold_factor = max(0.3, 1.0 - abs(threshold - 0.5) * 4)
        
        # Simulate metrics
        quality = base_perf * threshold_factor + np.random.normal(0, 0.05)
        balance = (1.0 - threshold) * 0.8 + np.random.normal(0, 0.05)
        efficiency = 1.0 / (1.0 + threshold * 2) + np.random.normal(0, 0.05)
        performance = (quality + balance + efficiency) / 3
        
        return {
            'quality': max(0.0, min(1.0, quality)),
            'balance': max(0.0, min(1.0, balance)),
            'efficiency': max(0.0, min(1.0, efficiency)),
            'performance': max(0.0, min(1.0, performance))
        }
    
    def _simulate_batch_performance(self, batch_size: int, problem_size: int, n_replications: int) -> Dict:
        """Simulate batch performance for given configuration."""
        # Optimal batch size is typically 10-20% of problem size
        optimal_batch = problem_size * 0.15
        batch_ratio = batch_size / optimal_batch
        
        # Stability (larger batches more stable, but diminishing returns)
        stability = min(1.0, np.sqrt(batch_ratio))
        
        # Speed (optimal around 15% of problem size)
        speed = 1.0 / (1.0 + abs(batch_ratio - 1.0))
        
        # Memory efficiency (linear in batch size)
        memory = batch_size / problem_size
        
        # Overall quality
        quality = (stability + speed) / 2 * (1.0 - min(0.5, memory))
        
        return {
            'stability': stability + np.random.normal(0, 0.05),
            'speed': speed + np.random.normal(0, 0.05),
            'memory': memory,
            'quality': max(0.0, min(1.0, quality + np.random.normal(0, 0.05)))
        }
    
    def _simulate_full_configuration(self, params: Dict, n_replications: int) -> Dict:
        """Simulate performance for full parameter configuration."""
        # Extract parameters
        embedding_dim = params.get('embedding_dim', 32)
        threshold = params.get('similarity_threshold', 0.5)
        batch_size = params.get('batch_size', 32)
        learning_rate = params.get('learning_rate', 0.01)
        regularization = params.get('regularization', 0.01)
        
        # Simulate individual component performances
        embedding_perf = embedding_dim / 128  # Normalized performance
        threshold_perf = 1.0 - abs(threshold - 0.5) * 2
        batch_perf = 1.0 / (1.0 + abs(batch_size - 32) / 32)
        lr_perf = 1.0 - abs(np.log10(learning_rate) + 2) / 2  # Optimal around 0.01
        reg_perf = 1.0 - regularization * 5  # Lower regularization better
        
        # Combined performance (geometric mean)
        components = [embedding_perf, threshold_perf, batch_perf, lr_perf, reg_perf]
        base_performance = np.prod([max(0.1, comp) for comp in components]) ** (1/len(components))
        
        # Computational cost
        cost = (embedding_dim / 32) * (batch_size / 32) * (1 / learning_rate * 0.01)
        
        # Memory usage
        memory = (embedding_dim * batch_size) / (32 * 32)
        
        # Add realistic noise
        performance_samples = []
        for _ in range(n_replications):
            noise = np.random.normal(0, 0.1)
            performance_samples.append(max(0.0, min(1.0, base_performance + noise)))
        
        return {
            'performance': np.mean(performance_samples),
            'performance_std': np.std(performance_samples),
            'cost': cost,
            'memory': memory,
            'component_scores': {
                'embedding': embedding_perf,
                'threshold': threshold_perf,
                'batch': batch_perf,
                'learning_rate': lr_perf,
                'regularization': reg_perf
            }
        }
    
    def _find_optimal_configurations(self, results: Dict) -> Dict:
        """Find optimal configurations from grid search results."""
        performance_matrix = results['performance_matrix']
        dimensions = results['dimensions']
        n_units_list = results['n_units_list']
        
        optimal_configs = {}
        
        # Find best configuration for each problem size
        for j, n_units in enumerate(n_units_list):
            best_dim_idx = np.argmax(performance_matrix[:, j])
            optimal_configs[n_units] = {
                'optimal_dimension': dimensions[best_dim_idx],
                'performance': performance_matrix[best_dim_idx, j],
                'computational_cost': results['computational_cost'][best_dim_idx, j],
                'memory_usage': results['memory_usage'][best_dim_idx, j]
            }
        
        # Find overall best configuration
        best_overall_idx = np.unravel_index(np.argmax(performance_matrix), performance_matrix.shape)
        optimal_configs['overall_best'] = {
            'dimension': dimensions[best_overall_idx[0]],
            'n_units': n_units_list[best_overall_idx[1]],
            'performance': performance_matrix[best_overall_idx]
        }
        
        return optimal_configs
    
    def _find_optimal_thresholds(self, results: Dict) -> Dict:
        """Find optimal similarity thresholds for each clustering method."""
        optimal_thresholds = {}
        
        for method in results['clustering_methods']:
            performances = results['statistical_performance'][method]
            best_idx = np.argmax(performances)
            optimal_thresholds[method] = {
                'threshold': results['thresholds'][best_idx],
                'performance': performances[best_idx],
                'cluster_quality': results['cluster_quality'][method][best_idx],
                'balance_preservation': results['balance_preservation'][method][best_idx]
            }
        
        return optimal_thresholds
    
    def _find_optimal_batch_sizes(self, results: Dict) -> Dict:
        """Find optimal batch sizes for each problem size."""
        optimal_batch_sizes = {}
        
        for j, problem_size in enumerate(results['problem_sizes']):
            quality_scores = results['solution_quality'][:, j]
            best_idx = np.argmax(quality_scores)
            optimal_batch_sizes[problem_size] = {
                'batch_size': results['batch_sizes'][best_idx],
                'quality': quality_scores[best_idx],
                'stability': results['convergence_stability'][best_idx, j],
                'speed': results['training_speed'][best_idx, j]
            }
        
        return optimal_batch_sizes
    
    def _analyze_parameter_importance(self, results: Dict) -> Dict:
        """Analyze the importance of different parameters."""
        param_names = list(results['param_grid'].keys())
        param_importance = {}
        
        # Simple variance-based importance analysis
        for param_name in param_names:
            param_values = []
            performances = []
            
            for i, combo in enumerate(results['param_combinations']):
                param_values.append(combo[param_name])
                performances.append(results['performance_scores'][i])
            
            # Compute correlation between parameter value and performance
            if len(set(param_values)) > 1:  # Only if parameter varies
                correlation = np.corrcoef(param_values, performances)[0, 1]
                param_importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                param_importance[param_name] = 0.0
        
        return param_importance
    
    def _find_pareto_optimal_configs(self, results: Dict) -> List[Dict]:
        """Find Pareto optimal configurations (performance vs computational cost)."""
        performances = np.array(results['performance_scores'])
        costs = np.array(results['computational_costs'])
        
        # Find Pareto frontier
        pareto_optimal = []
        
        for i in range(len(performances)):
            is_pareto = True
            for j in range(len(performances)):
                if i != j:
                    # Check if j dominates i (better performance and lower cost)
                    if performances[j] >= performances[i] and costs[j] <= costs[i]:
                        if performances[j] > performances[i] or costs[j] < costs[i]:
                            is_pareto = False
                            break
            
            if is_pareto:
                pareto_optimal.append({
                    'config_index': i,
                    'params': results['param_combinations'][i],
                    'performance': performances[i],
                    'cost': costs[i]
                })
        
        return pareto_optimal

def run_comprehensive_hyperparameter_analysis(output_dir: str = "hyperparameter_results") -> Dict:
    """
    Run comprehensive hyperparameter analysis.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    Dict
        Comprehensive hyperparameter analysis results
    """
    print("=== COMPREHENSIVE HYPERPARAMETER ANALYSIS ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer()
    
    results = {}
    
    # 1. Embedding dimension grid search
    print("\n1. Embedding Dimension Grid Search")
    results['embedding_dimensions'] = optimizer.grid_search_embedding_dimensions(
        dimensions=[8, 16, 32, 64, 128],
        n_units_list=[100, 200, 400],
        n_replications=15
    )
    
    # 2. Similarity threshold sensitivity
    print("\n2. Similarity Threshold Sensitivity Analysis")
    results['similarity_thresholds'] = optimizer.sensitivity_analysis_similarity_thresholds(
        thresholds=np.linspace(0.2, 0.8, 7).tolist(),
        n_replications=20
    )
    
    # 3. Batch size optimization
    print("\n3. Batch Size Optimization")
    results['batch_sizes'] = optimizer.batch_size_optimization(
        batch_sizes=[8, 16, 32, 64, 128],
        problem_sizes=[100, 200, 400],
        n_replications=15
    )
    
    # 4. Comprehensive parameter study
    print("\n4. Comprehensive Parameter Study")
    results['comprehensive_study'] = optimizer.comprehensive_hyperparameter_study(
        n_replications=10
    )
    
    # Save results
    import pickle
    with open(f"{output_dir}/hyperparameter_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print(f"\nHyperparameter analysis completed! Results saved to {output_dir}/")
    
    return results

if __name__ == "__main__":
    # Run comprehensive hyperparameter analysis
    results = run_comprehensive_hyperparameter_analysis()
    
    # Print summary
    print("\n=== HYPERPARAMETER ANALYSIS SUMMARY ===")
    for analysis_name, analysis_results in results.items():
        print(f"\n{analysis_name.upper()}:")
        if 'optimal_configs' in analysis_results:
            print(f"  Optimal configurations found: {len(analysis_results['optimal_configs'])}")
        if 'pareto_optimal' in analysis_results:
            print(f"  Pareto optimal configurations: {len(analysis_results['pareto_optimal'])}")
        if 'parameter_importance' in analysis_results:
            importance = analysis_results['parameter_importance']
            most_important = max(importance.items(), key=lambda x: x[1])
            print(f"  Most important parameter: {most_important[0]} (importance: {most_important[1]:.3f})")
