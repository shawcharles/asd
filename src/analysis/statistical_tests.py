"""Statistical testing framework for method comparison analysis.

This module provides comprehensive statistical testing capabilities for comparing
the performance of different experimental design methods, including significance
tests, confidence intervals, effect sizes, and power analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power
import warnings


@dataclass
class ComparisonResult:
    """Results of a statistical comparison between two methods."""
    method1: str
    method2: str
    statistic: float
    p_value: float
    p_value_corrected: Optional[float]
    effect_size: float
    confidence_interval: Tuple[float, float]
    test_type: str
    significant: bool
    practical_significance: bool


@dataclass
class BootstrapResult:
    """Results of bootstrap confidence interval calculation."""
    statistic: float
    confidence_interval: Tuple[float, float]
    bootstrap_samples: np.ndarray
    confidence_level: float


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for method comparison studies."""
    
    def __init__(self, alpha: float = 0.05, practical_threshold: float = 0.01):
        """Initialize the statistical analyzer.
        
        Args:
            alpha: Significance level for statistical tests
            practical_threshold: Threshold for practical significance (e.g., 1% bias reduction)
        """
        self.alpha = alpha
        self.practical_threshold = practical_threshold
    
    def paired_comparison_test(
        self, 
        method1_results: np.ndarray, 
        method2_results: np.ndarray,
        method1_name: str = "Method1",
        method2_name: str = "Method2",
        test_type: str = "auto"
    ) -> ComparisonResult:
        """Perform paired comparison test between two methods.
        
        Args:
            method1_results: Results from first method
            method2_results: Results from second method  
            method1_name: Name of first method
            method2_name: Name of second method
            test_type: Type of test ('ttest', 'wilcoxon', 'auto')
            
        Returns:
            ComparisonResult object with test statistics and interpretation
        """
        if len(method1_results) != len(method2_results):
            raise ValueError("Method results must have same length for paired comparison")
        
        # Calculate differences
        differences = method1_results - method2_results
        
        # Choose test type
        if test_type == "auto":
            # Use Shapiro-Wilk test to check normality of differences
            if len(differences) >= 8:  # Minimum for Shapiro-Wilk
                _, p_normal = stats.shapiro(differences)
                test_type = "ttest" if p_normal > 0.05 else "wilcoxon"
            else:
                test_type = "wilcoxon"  # Conservative choice for small samples
        
        # Perform statistical test
        if test_type == "ttest":
            statistic, p_value = stats.ttest_rel(method1_results, method2_results)
        elif test_type == "wilcoxon":
            statistic, p_value = stats.wilcoxon(method1_results, method2_results, 
                                              alternative='two-sided', zero_method='zsplit')
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Calculate effect size (Cohen's d for paired samples)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        
        # Calculate confidence interval for the difference
        ci = self._paired_difference_ci(differences)
        
        # Determine significance
        significant = p_value < self.alpha
        practical_significance = abs(np.mean(differences)) > self.practical_threshold
        
        return ComparisonResult(
            method1=method1_name,
            method2=method2_name,
            statistic=statistic,
            p_value=p_value,
            p_value_corrected=None,  # Will be set by multiple_comparison_correction
            effect_size=effect_size,
            confidence_interval=ci,
            test_type=test_type,
            significant=significant,
            practical_significance=practical_significance
        )
    
    def multiple_comparison_correction(
        self, 
        comparison_results: List[ComparisonResult], 
        method: str = 'bonferroni'
    ) -> List[ComparisonResult]:
        """Apply multiple comparison correction to a list of comparison results.
        
        Args:
            comparison_results: List of ComparisonResult objects
            method: Correction method ('bonferroni', 'holm', 'fdr_bh')
            
        Returns:
            Updated list of ComparisonResult objects with corrected p-values
        """
        p_values = [result.p_value for result in comparison_results]
        
        # Apply correction
        rejected, p_corrected, _, _ = multipletests(p_values, alpha=self.alpha, method=method)
        
        # Update results
        corrected_results = []
        for i, result in enumerate(comparison_results):
            updated_result = ComparisonResult(
                method1=result.method1,
                method2=result.method2,
                statistic=result.statistic,
                p_value=result.p_value,
                p_value_corrected=p_corrected[i],
                effect_size=result.effect_size,
                confidence_interval=result.confidence_interval,
                test_type=result.test_type,
                significant=rejected[i],  # Updated based on corrected p-value
                practical_significance=result.practical_significance
            )
            corrected_results.append(updated_result)
        
        return corrected_results
    
    def effect_size_calculation(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray, 
        effect_type: str = "cohen_d"
    ) -> float:
        """Calculate effect size between two groups.
        
        Args:
            group1: First group of observations
            group2: Second group of observations
            effect_type: Type of effect size ('cohen_d', 'hedges_g', 'glass_delta')
            
        Returns:
            Effect size value
        """
        mean1, mean2 = np.mean(group1), np.mean(group2)
        
        if effect_type == "cohen_d":
            # Pooled standard deviation
            n1, n2 = len(group1), len(group2)
            pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
            return (mean1 - mean2) / pooled_std
        
        elif effect_type == "hedges_g":
            # Hedges' g (bias-corrected Cohen's d)
            cohen_d = self.effect_size_calculation(group1, group2, "cohen_d")
            n = len(group1) + len(group2)
            correction = 1 - (3 / (4*n - 9))
            return cohen_d * correction
        
        elif effect_type == "glass_delta":
            # Glass's delta (uses control group SD)
            return (mean1 - mean2) / np.std(group2, ddof=1)
        
        else:
            raise ValueError(f"Unknown effect size type: {effect_type}")
    
    def bootstrap_confidence_interval(
        self, 
        data: np.ndarray, 
        statistic: callable = np.mean,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = None
    ) -> BootstrapResult:
        """Calculate bootstrap confidence interval for a statistic.
        
        Args:
            data: Input data array
            statistic: Function to calculate statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            random_seed: Random seed for reproducibility
            
        Returns:
            BootstrapResult object with CI and bootstrap samples
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Calculate original statistic
        original_stat = statistic(data)
        
        # Generate bootstrap samples
        n = len(data)
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return BootstrapResult(
            statistic=original_stat,
            confidence_interval=(ci_lower, ci_upper),
            bootstrap_samples=bootstrap_stats,
            confidence_level=confidence_level
        )
    
    def power_analysis(
        self, 
        effect_size: float, 
        sample_size: Optional[int] = None,
        alpha: Optional[float] = None,
        power: Optional[float] = None
    ) -> Dict[str, float]:
        """Perform power analysis for t-test.
        
        Provide three of the four parameters to calculate the fourth.
        
        Args:
            effect_size: Cohen's d effect size
            sample_size: Sample size per group
            alpha: Significance level
            power: Statistical power
            
        Returns:
            Dictionary with all four parameters
        """
        alpha = alpha or self.alpha
        
        # Count non-None parameters
        params = [sample_size, alpha, power]
        n_provided = sum(p is not None for p in params)
        
        if n_provided != 2:
            raise ValueError("Must provide exactly 2 of: sample_size, alpha, power")
        
        if sample_size is None:
            # Calculate required sample size
            from statsmodels.stats.power import tt_solve_power
            sample_size = tt_solve_power(effect_size=effect_size, alpha=alpha, power=power)
            sample_size = int(np.ceil(sample_size))
        
        elif power is None:
            # Calculate achieved power
            power = ttest_power(effect_size, sample_size, alpha)
        
        elif alpha is None:
            # Calculate required alpha (less common)
            from statsmodels.stats.power import tt_solve_power
            alpha = tt_solve_power(effect_size=effect_size, nobs=sample_size, power=power)
        
        return {
            'effect_size': effect_size,
            'sample_size': sample_size,
            'alpha': alpha,
            'power': power
        }
    
    def comprehensive_method_comparison(
        self, 
        results_df: pd.DataFrame,
        metric_column: str = 'abs_bias',
        method_column: str = 'method',
        seed_column: str = 'seed'
    ) -> Dict[str, Any]:
        """Perform comprehensive statistical comparison of multiple methods.
        
        Args:
            results_df: DataFrame with results from different methods
            metric_column: Column name containing the metric to compare
            method_column: Column name containing method names
            seed_column: Column name containing seed/replicate identifiers
            
        Returns:
            Dictionary with comprehensive comparison results
        """
        methods = results_df[method_column].unique()
        n_methods = len(methods)
        
        # Prepare data for pairwise comparisons
        method_data = {}
        for method in methods:
            method_data[method] = results_df[results_df[method_column] == method][metric_column].values
        
        # Perform pairwise comparisons
        comparisons = []
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j:  # Avoid duplicate comparisons
                    comparison = self.paired_comparison_test(
                        method_data[method1], 
                        method_data[method2],
                        method1, 
                        method2
                    )
                    comparisons.append(comparison)
        
        # Apply multiple comparison correction
        corrected_comparisons = self.multiple_comparison_correction(comparisons)
        
        # Calculate summary statistics for each method
        summary_stats = {}
        for method in methods:
            data = method_data[method]
            bootstrap_result = self.bootstrap_confidence_interval(data)
            
            summary_stats[method] = {
                'mean': np.mean(data),
                'std': np.std(data, ddof=1),
                'median': np.median(data),
                'ci_lower': bootstrap_result.confidence_interval[0],
                'ci_upper': bootstrap_result.confidence_interval[1],
                'n_observations': len(data)
            }
        
        # Overall analysis summary
        analysis_summary = {
            'n_methods': n_methods,
            'n_comparisons': len(comparisons),
            'significant_comparisons': sum(comp.significant for comp in corrected_comparisons),
            'practically_significant_comparisons': sum(comp.practical_significance for comp in corrected_comparisons),
            'alpha_level': self.alpha,
            'practical_threshold': self.practical_threshold
        }
        
        return {
            'pairwise_comparisons': corrected_comparisons,
            'summary_statistics': summary_stats,
            'analysis_summary': analysis_summary
        }
    
    def _paired_difference_ci(self, differences: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for paired differences."""
        n = len(differences)
        mean_diff = np.mean(differences)
        se_diff = np.std(differences, ddof=1) / np.sqrt(n)
        
        # Use t-distribution for small samples
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n-1)
        
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        return (ci_lower, ci_upper)
    
    def format_comparison_results(self, comparisons: List[ComparisonResult]) -> pd.DataFrame:
        """Format comparison results as a readable DataFrame.
        
        Args:
            comparisons: List of ComparisonResult objects
            
        Returns:
            Formatted DataFrame with comparison results
        """
        data = []
        for comp in comparisons:
            data.append({
                'Method 1': comp.method1,
                'Method 2': comp.method2,
                'Test Statistic': f"{comp.statistic:.4f}",
                'P-value': f"{comp.p_value:.4f}",
                'P-value (Corrected)': f"{comp.p_value_corrected:.4f}" if comp.p_value_corrected else "N/A",
                'Effect Size (Cohen\'s d)': f"{comp.effect_size:.4f}",
                'CI Lower': f"{comp.confidence_interval[0]:.4f}",
                'CI Upper': f"{comp.confidence_interval[1]:.4f}",
                'Test Type': comp.test_type,
                'Statistically Significant': "Yes" if comp.significant else "No",
                'Practically Significant': "Yes" if comp.practical_significance else "No"
            })
        
        return pd.DataFrame(data)


def interpret_effect_size(effect_size: float) -> str:
    """Interpret Cohen's d effect size according to conventional thresholds.
    
    Args:
        effect_size: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_effect = abs(effect_size)
    
    if abs_effect < 0.2:
        return "Negligible"
    elif abs_effect < 0.5:
        return "Small"
    elif abs_effect < 0.8:
        return "Medium"
    else:
        return "Large"


def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
    """Provide interpretation of p-value.
    
    Args:
        p_value: P-value from statistical test
        alpha: Significance threshold
        
    Returns:
        Interpretation string
    """
    if p_value < 0.001:
        return "Highly significant (p < 0.001)"
    elif p_value < 0.01:
        return "Very significant (p < 0.01)"
    elif p_value < alpha:
        return f"Significant (p < {alpha})"
    elif p_value < 0.1:
        return "Marginally significant (p < 0.1)"
    else:
        return "Not significant"
