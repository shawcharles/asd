"""Publication-ready plotting functions with statistical annotations.

This module provides enhanced plotting capabilities for the Adaptive Supergeo Design
paper, including confidence intervals, significance indicators, and professional
formatting suitable for academic publication.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import warnings
from src.analysis.statistical_tests import StatisticalAnalyzer, ComparisonResult


# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Publication settings
PUBLICATION_CONFIG = {
    'figure_size': (10, 6),
    'dpi': 300,
    'font_size': 12,
    'title_size': 14,
    'label_size': 12,
    'legend_size': 10,
    'line_width': 2,
    'marker_size': 8,
    'capsize': 5,
    'capthick': 2,
    'alpha': 0.7,
    'colors': {
        'TM-Baseline': '#1f77b4',
        'SG-TM': '#ff7f0e',
        'ASD-TM': '#2ca02c',
        'ASD': '#2ca02c',
        'TM': '#1f77b4',
        'SG': '#ff7f0e'
    }
}


class PublicationPlotter:
    """Publication-ready plotting class with statistical annotations."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize plotter with configuration.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = PUBLICATION_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Initialize statistical analyzer
        self.analyzer = StatisticalAnalyzer()
        
        # Set matplotlib parameters
        plt.rcParams.update({
            'font.size': self.config['font_size'],
            'axes.titlesize': self.config['title_size'],
            'axes.labelsize': self.config['label_size'],
            'legend.fontsize': self.config['legend_size'],
            'lines.linewidth': self.config['line_width'],
            'lines.markersize': self.config['marker_size']
        })
    
    def bias_comparison_plot(
        self, 
        results_df: pd.DataFrame,
        metric_col: str = 'abs_bias',
        method_col: str = 'method',
        include_ci: bool = True,
        include_significance: bool = True,
        save_path: Optional[str] = None,
        title: str = "Distribution of |Bias| across Methods"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create bias comparison plot with confidence intervals and significance tests.
        
        Args:
            results_df: DataFrame with results
            metric_col: Column name for the metric to plot
            method_col: Column name for method identification
            include_ci: Whether to include confidence intervals
            include_significance: Whether to include significance annotations
            save_path: Optional path to save the figure
            title: Plot title
            
        Returns:
            Tuple of (figure, axes) objects
        """
        fig, ax = plt.subplots(figsize=self.config['figure_size'], dpi=self.config['dpi'])
        
        # Get unique methods and prepare data
        methods = results_df[method_col].unique()
        method_data = {}
        for method in methods:
            method_data[method] = results_df[results_df[method_col] == method][metric_col].values
        
        # Create box plot
        positions = range(len(methods))
        box_data = [method_data[method] for method in methods]
        
        bp = ax.boxplot(box_data, positions=positions, patch_artist=True, 
                       widths=0.6, showfliers=True, notch=True)
        
        # Color boxes according to method
        for i, (patch, method) in enumerate(zip(bp['boxes'], methods)):
            color = self.config['colors'].get(method, f'C{i}')
            patch.set_facecolor(color)
            patch.set_alpha(self.config['alpha'])
        
        # Add confidence intervals if requested
        if include_ci:
            for i, method in enumerate(methods):
                data = method_data[method]
                bootstrap_result = self.analyzer.bootstrap_confidence_interval(data)
                ci_lower, ci_upper = bootstrap_result.confidence_interval
                
                # Add CI as error bars
                ax.errorbar(i, bootstrap_result.statistic, 
                           yerr=[[bootstrap_result.statistic - ci_lower], 
                                 [ci_upper - bootstrap_result.statistic]],
                           fmt='D', color='red', capsize=self.config['capsize'], 
                           capthick=self.config['capthick'], markersize=6)
        
        # Add significance annotations if requested
        if include_significance and len(methods) > 1:
            comparison_results = self.analyzer.comprehensive_method_comparison(
                results_df, metric_col, method_col
            )
            self._add_significance_annotations(ax, comparison_results['pairwise_comparisons'], 
                                             methods, positions)
        
        # Formatting
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel(f'{metric_col.replace("_", " ").title()}')
        ax.set_title(title, fontsize=self.config['title_size'], pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add legend for CI if included
        if include_ci:
            ci_patch = mpatches.Patch(color='red', label='95% Bootstrap CI')
            ax.legend(handles=[ci_patch], loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        
        return fig, ax
    
    def rmse_comparison_plot(
        self, 
        results_df: pd.DataFrame,
        metric_col: str = 'rmse',
        method_col: str = 'method',
        include_significance: bool = True,
        save_path: Optional[str] = None,
        title: str = "RMSE Comparison Across Methods"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create RMSE comparison bar plot with confidence intervals.
        
        Args:
            results_df: DataFrame with results
            metric_col: Column name for RMSE values
            method_col: Column name for method identification
            include_significance: Whether to include significance tests
            save_path: Optional path to save the figure
            title: Plot title
            
        Returns:
            Tuple of (figure, axes) objects
        """
        fig, ax = plt.subplots(figsize=self.config['figure_size'], dpi=self.config['dpi'])
        
        # Calculate summary statistics
        summary_stats = results_df.groupby(method_col)[metric_col].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        # Calculate confidence intervals
        summary_stats['ci_lower'] = summary_stats['mean'] - 1.96 * summary_stats['std'] / np.sqrt(summary_stats['count'])
        summary_stats['ci_upper'] = summary_stats['mean'] + 1.96 * summary_stats['std'] / np.sqrt(summary_stats['count'])
        summary_stats['ci_error'] = 1.96 * summary_stats['std'] / np.sqrt(summary_stats['count'])
        
        # Create bar plot
        methods = summary_stats[method_col].values
        means = summary_stats['mean'].values
        errors = summary_stats['ci_error'].values
        
        colors = [self.config['colors'].get(method, f'C{i}') for i, method in enumerate(methods)]
        
        bars = ax.bar(methods, means, yerr=errors, capsize=self.config['capsize'],
                     color=colors, alpha=self.config['alpha'], 
                     error_kw={'capthick': self.config['capthick']})
        
        # Add value labels on bars
        for bar, mean, error in zip(bars, means, errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.001,
                   f'{mean:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Add significance annotations if requested
        if include_significance and len(methods) > 1:
            comparison_results = self.analyzer.comprehensive_method_comparison(
                results_df, metric_col, method_col
            )
            self._add_bar_significance_annotations(ax, comparison_results['pairwise_comparisons'], 
                                                 methods, means + errors)
        
        # Formatting
        ax.set_ylabel(f'{metric_col.upper()}')
        ax.set_title(title, fontsize=self.config['title_size'], pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        
        return fig, ax
    
    def scalability_plot(
        self, 
        scalability_df: pd.DataFrame,
        n_col: str = 'N',
        runtime_col: str = 'runtime_sec',
        method_col: str = 'method',
        include_regression: bool = True,
        save_path: Optional[str] = None,
        title: str = "Design Optimization Runtime vs N"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create scalability plot with regression lines and confidence intervals.
        
        Args:
            scalability_df: DataFrame with scalability results
            n_col: Column name for N (problem size)
            runtime_col: Column name for runtime
            method_col: Column name for method identification
            include_regression: Whether to include regression lines
            save_path: Optional path to save the figure
            title: Plot title
            
        Returns:
            Tuple of (figure, axes) objects
        """
        fig, ax = plt.subplots(figsize=self.config['figure_size'], dpi=self.config['dpi'])
        
        methods = scalability_df[method_col].unique()
        
        for i, method in enumerate(methods):
            method_data = scalability_df[scalability_df[method_col] == method]
            n_values = method_data[n_col].values
            runtime_values = method_data[runtime_col].values
            
            color = self.config['colors'].get(method, f'C{i}')
            
            # Plot data points
            ax.loglog(n_values, runtime_values, 'o-', label=method, 
                     color=color, linewidth=self.config['line_width'],
                     markersize=self.config['marker_size'])
            
            # Add regression line if requested
            if include_regression and len(n_values) > 2:
                # Fit log-log regression
                log_n = np.log(n_values)
                log_runtime = np.log(runtime_values)
                
                # Linear regression in log space
                coeffs = np.polyfit(log_n, log_runtime, 1)
                slope, intercept = coeffs
                
                # Generate smooth curve for plotting
                n_smooth = np.logspace(np.log10(n_values.min()), 
                                      np.log10(n_values.max()), 100)
                runtime_smooth = np.exp(intercept) * n_smooth**slope
                
                ax.loglog(n_smooth, runtime_smooth, '--', color=color, 
                         alpha=0.7, linewidth=1.5)
                
                # Add complexity annotation
                ax.text(n_values[-1], runtime_values[-1], 
                       f'O(N^{slope:.2f})', 
                       fontsize=10, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
        
        # Formatting
        ax.set_xlabel(f'Number of geos ({n_col})')
        ax.set_ylabel(f'Wall-clock runtime (s, log scale)')
        ax.set_title(title, fontsize=self.config['title_size'], pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        
        return fig, ax
    
    def robustness_heatmap(
        self, 
        robustness_results: pd.DataFrame,
        metric_col: str = 'performance_ratio',
        scenario_col: str = 'scenario',
        method_col: str = 'method',
        save_path: Optional[str] = None,
        title: str = "Method Robustness Across Scenarios"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create robustness heatmap showing method performance across scenarios.
        
        Args:
            robustness_results: DataFrame with robustness test results
            metric_col: Column name for performance metric
            scenario_col: Column name for scenario identification
            method_col: Column name for method identification
            save_path: Optional path to save the figure
            title: Plot title
            
        Returns:
            Tuple of (figure, axes) objects
        """
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.config['dpi'])
        
        # Pivot data for heatmap
        heatmap_data = robustness_results.pivot(index=scenario_col, 
                                               columns=method_col, 
                                               values=metric_col)
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=1.0, ax=ax, cbar_kws={'label': 'Performance Ratio'})
        
        # Formatting
        ax.set_title(title, fontsize=self.config['title_size'], pad=20)
        ax.set_xlabel('Method')
        ax.set_ylabel('Robustness Scenario')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        
        return fig, ax
    
    def _add_significance_annotations(
        self, 
        ax: plt.Axes, 
        comparisons: List[ComparisonResult],
        methods: List[str],
        positions: List[int]
    ) -> None:
        """Add significance annotations to box plot."""
        method_to_pos = {method: pos for method, pos in zip(methods, positions)}
        
        # Find maximum y value for positioning annotations
        y_max = ax.get_ylim()[1]
        y_offset = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        
        annotation_height = y_max
        for i, comp in enumerate(comparisons):
            if comp.significant:
                pos1 = method_to_pos[comp.method1]
                pos2 = method_to_pos[comp.method2]
                
                # Draw significance bar
                annotation_height += y_offset
                ax.plot([pos1, pos2], [annotation_height, annotation_height], 'k-', linewidth=1)
                ax.plot([pos1, pos1], [annotation_height - y_offset/3, annotation_height + y_offset/3], 'k-', linewidth=1)
                ax.plot([pos2, pos2], [annotation_height - y_offset/3, annotation_height + y_offset/3], 'k-', linewidth=1)
                
                # Add significance symbol
                if comp.p_value_corrected < 0.001:
                    sig_symbol = '***'
                elif comp.p_value_corrected < 0.01:
                    sig_symbol = '**'
                elif comp.p_value_corrected < 0.05:
                    sig_symbol = '*'
                else:
                    sig_symbol = 'ns'
                
                ax.text((pos1 + pos2) / 2, annotation_height + y_offset/4, sig_symbol,
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Adjust y-axis limits to accommodate annotations
        if annotation_height > y_max:
            ax.set_ylim(ax.get_ylim()[0], annotation_height + y_offset)
    
    def _add_bar_significance_annotations(
        self, 
        ax: plt.Axes, 
        comparisons: List[ComparisonResult],
        methods: List[str],
        bar_heights: np.ndarray
    ) -> None:
        """Add significance annotations to bar plot."""
        method_to_idx = {method: i for i, method in enumerate(methods)}
        
        # Find maximum bar height for positioning annotations
        max_height = np.max(bar_heights)
        y_offset = 0.05 * max_height
        
        annotation_height = max_height
        for comp in comparisons:
            if comp.significant:
                idx1 = method_to_idx[comp.method1]
                idx2 = method_to_idx[comp.method2]
                
                # Draw significance bar
                annotation_height += y_offset
                ax.plot([idx1, idx2], [annotation_height, annotation_height], 'k-', linewidth=1)
                ax.plot([idx1, idx1], [annotation_height - y_offset/3, annotation_height + y_offset/3], 'k-', linewidth=1)
                ax.plot([idx2, idx2], [annotation_height - y_offset/3, annotation_height + y_offset/3], 'k-', linewidth=1)
                
                # Add significance symbol
                if comp.p_value_corrected < 0.001:
                    sig_symbol = '***'
                elif comp.p_value_corrected < 0.01:
                    sig_symbol = '**'
                elif comp.p_value_corrected < 0.05:
                    sig_symbol = '*'
                else:
                    sig_symbol = 'ns'
                
                ax.text((idx1 + idx2) / 2, annotation_height + y_offset/4, sig_symbol,
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Adjust y-axis limits to accommodate annotations
        if annotation_height > max_height:
            ax.set_ylim(ax.get_ylim()[0], annotation_height + y_offset)
    
    def create_summary_table_plot(
        self, 
        comparison_results: Dict[str, Any],
        save_path: Optional[str] = None,
        title: str = "Statistical Comparison Summary"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a summary table plot of statistical comparison results.
        
        Args:
            comparison_results: Results from comprehensive_method_comparison
            save_path: Optional path to save the figure
            title: Plot title
            
        Returns:
            Tuple of (figure, axes) objects
        """
        fig, ax = plt.subplots(figsize=(14, 8), dpi=self.config['dpi'])
        ax.axis('tight')
        ax.axis('off')
        
        # Format pairwise comparisons
        comparisons_df = self.analyzer.format_comparison_results(
            comparison_results['pairwise_comparisons']
        )
        
        # Create table
        table = ax.table(cellText=comparisons_df.values,
                        colLabels=comparisons_df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0.3, 1, 0.6])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color code significant results
        for i in range(len(comparisons_df)):
            if comparisons_df.iloc[i]['Statistically Significant'] == 'Yes':
                for j in range(len(comparisons_df.columns)):
                    table[(i+1, j)].set_facecolor('#90EE90')  # Light green
        
        # Add summary statistics below
        summary_text = f"""
        Analysis Summary:
        • Number of methods compared: {comparison_results['analysis_summary']['n_methods']}
        • Number of pairwise comparisons: {comparison_results['analysis_summary']['n_comparisons']}
        • Statistically significant comparisons: {comparison_results['analysis_summary']['significant_comparisons']}
        • Practically significant comparisons: {comparison_results['analysis_summary']['practically_significant_comparisons']}
        • Significance level (α): {comparison_results['analysis_summary']['alpha_level']}
        • Practical significance threshold: {comparison_results['analysis_summary']['practical_threshold']}
        """
        
        ax.text(0.5, 0.2, summary_text, transform=ax.transAxes, 
               fontsize=10, ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        ax.set_title(title, fontsize=self.config['title_size'], pad=20)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        
        return fig, ax


    def power_analysis_enhanced_plots(
        self,
        power_df: pd.DataFrame,
        output_dir: str = "plots/power_enhanced",
        save_plots: bool = True
    ) -> Dict[str, plt.Figure]:
        """Create enhanced power analysis plots with confidence intervals and significance indicators.
        
        Args:
            power_df: DataFrame with power analysis results
            output_dir: Directory to save plots
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary of figure objects
        """
        import os
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        
        figures = {}
        
        # 1. Power curves by effect size with confidence intervals
        effect_sizes = sorted(power_df['effect_size'].unique())
        
        for effect_size in effect_sizes:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=self.config['dpi'])
            
            subset = power_df[power_df['effect_size'] == effect_size]
            
            for method in subset['method'].unique():
                method_data = subset[subset['method'] == method]
                
                # Calculate confidence intervals for power estimates
                power_values = method_data['power'].values
                sample_sizes = method_data['sample_size'].values
                
                # Bootstrap confidence intervals for power
                ci_lower, ci_upper = [], []
                for power in power_values:
                    # Wilson score interval for binomial proportions
                    n = 100  # number of replications
                    z = 1.96  # 95% CI
                    p_hat = power
                    
                    denominator = 1 + z**2/n
                    centre = (p_hat + z**2/(2*n)) / denominator
                    margin = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n))/n) / denominator
                    
                    ci_lower.append(max(0, centre - margin))
                    ci_upper.append(min(1, centre + margin))
                
                # Plot with confidence intervals
                color = self.config['colors'].get(method, f'C{list(subset["method"].unique()).index(method)}')
                ax.plot(sample_sizes, power_values, 'o-', color=color, label=method, 
                       linewidth=self.config['line_width'], markersize=self.config['marker_size'])
                ax.fill_between(sample_sizes, ci_lower, ci_upper, alpha=0.2, color=color)
            
            # Add horizontal line at 80% power
            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Power Threshold')
            
            ax.set_xlabel('Sample Size (N)', fontsize=self.config['label_size'])
            ax.set_ylabel('Statistical Power', fontsize=self.config['label_size'])
            ax.set_title(f'Power Analysis: Effect Size = {effect_size}', fontsize=self.config['title_size'])
            ax.legend(fontsize=self.config['legend_size'])
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            
            plt.tight_layout()
            
            if save_plots:
                filename = f'power_curve_enhanced_es_{str(effect_size).replace(".", "p")}.png'
                fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=self.config['dpi'])
            
            figures[f'power_es_{effect_size}'] = fig
        
        # 2. Power heatmap across all conditions
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.config['dpi'])
        
        # Pivot data for heatmap
        for i, method in enumerate(power_df['method'].unique()):
            method_data = power_df[power_df['method'] == method]
            pivot_data = method_data.pivot(index='effect_size', columns='sample_size', values='power')
            
            plt.subplot(1, len(power_df['method'].unique()), i+1)
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                       vmin=0, vmax=1, cbar_kws={'label': 'Statistical Power'})
            plt.title(f'{method} Power Analysis', fontsize=self.config['title_size'])
            plt.xlabel('Sample Size', fontsize=self.config['label_size'])
            if i == 0:
                plt.ylabel('Effect Size', fontsize=self.config['label_size'])
            else:
                plt.ylabel('')
        
        plt.tight_layout()
        
        if save_plots:
            fig.savefig(os.path.join(output_dir, 'power_heatmap_comparison.png'), 
                       bbox_inches='tight', dpi=self.config['dpi'])
        
        figures['power_heatmap'] = fig
        
        return figures
    
    def efficiency_comparison_plots(
        self,
        efficiency_df: pd.DataFrame,
        output_dir: str = "plots/efficiency_enhanced",
        save_plots: bool = True
    ) -> Dict[str, plt.Figure]:
        """Create enhanced computational efficiency comparison plots.
        
        Args:
            efficiency_df: DataFrame with efficiency analysis results
            output_dir: Directory to save plots
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary of figure objects
        """
        import os
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        
        figures = {}
        
        # 1. Execution time comparison with error bars
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=self.config['dpi'])
        
        # Check available columns and map to expected names
        column_mapping = {
            'execution_time_seconds': 'execution_time',
            'memory_usage_mb': 'memory_usage',
            'time_per_unit_ms': 'time_per_unit'
        }
        
        # Rename columns if they exist
        efficiency_df_renamed = efficiency_df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in efficiency_df_renamed.columns:
                efficiency_df_renamed[new_col] = efficiency_df_renamed[old_col]
        
        # Filter out NaN values for plotting - check which columns actually exist
        available_cols = [col for col in ['execution_time', 'memory_usage'] if col in efficiency_df_renamed.columns]
        if available_cols:
            valid_data = efficiency_df_renamed.dropna(subset=available_cols)
        else:
            valid_data = pd.DataFrame()  # Empty DataFrame if no valid columns
        
        if not valid_data.empty:
            # Execution time plot
            for method in valid_data['method'].unique():
                method_data = valid_data[valid_data['method'] == method]
                color = self.config['colors'].get(method, f'C{list(valid_data["method"].unique()).index(method)}')
                
                ax1.plot(method_data['sample_size'], method_data['execution_time'], 
                        'o-', color=color, label=method, linewidth=self.config['line_width'])
            
            ax1.set_xlabel('Sample Size', fontsize=self.config['label_size'])
            ax1.set_ylabel('Execution Time (seconds)', fontsize=self.config['label_size'])
            ax1.set_title('Execution Time Scalability', fontsize=self.config['title_size'])
            ax1.legend(fontsize=self.config['legend_size'])
            ax1.grid(True, alpha=0.3)
            
            # Memory usage plot
            for method in valid_data['method'].unique():
                method_data = valid_data[valid_data['method'] == method]
                color = self.config['colors'].get(method, f'C{list(valid_data["method"].unique()).index(method)}')
                
                ax2.plot(method_data['sample_size'], method_data['memory_usage'], 
                        'o-', color=color, label=method, linewidth=self.config['line_width'])
            
            ax2.set_xlabel('Sample Size', fontsize=self.config['label_size'])
            ax2.set_ylabel('Memory Usage (MB)', fontsize=self.config['label_size'])
            ax2.set_title('Memory Usage Scalability', fontsize=self.config['title_size'])
            ax2.legend(fontsize=self.config['legend_size'])
            ax2.grid(True, alpha=0.3)
            
            # Time per unit plot
            for method in valid_data['method'].unique():
                method_data = valid_data[valid_data['method'] == method]
                if 'time_per_unit' in method_data.columns:
                    color = self.config['colors'].get(method, f'C{list(valid_data["method"].unique()).index(method)}')
                    
                    ax3.plot(method_data['sample_size'], method_data['time_per_unit'], 
                            'o-', color=color, label=method, linewidth=self.config['line_width'])
            
            ax3.set_xlabel('Sample Size', fontsize=self.config['label_size'])
            ax3.set_ylabel('Time per Unit (ms)', fontsize=self.config['label_size'])
            ax3.set_title('Computational Complexity', fontsize=self.config['title_size'])
            ax3.legend(fontsize=self.config['legend_size'])
            ax3.grid(True, alpha=0.3)
        else:
            # Handle case with no valid data
            for ax in [ax1, ax2, ax3]:
                ax.text(0.5, 0.5, 'No valid efficiency data available', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_plots:
            fig.savefig(os.path.join(output_dir, 'efficiency_comparison_enhanced.png'), 
                       bbox_inches='tight', dpi=self.config['dpi'])
        
        figures['efficiency_comparison'] = fig
        
        return figures
    
    def generate_results_tables(
        self,
        power_df: pd.DataFrame,
        efficiency_df: pd.DataFrame,
        output_dir: str = "tables",
        format_type: str = "latex"
    ) -> Dict[str, str]:
        """Generate comprehensive results tables for publication.
        
        Args:
            power_df: DataFrame with power analysis results
            efficiency_df: DataFrame with efficiency analysis results
            output_dir: Directory to save tables
            format_type: Format type ('latex', 'html', 'csv')
            
        Returns:
            Dictionary mapping table names to file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        table_paths = {}
        
        # 1. Power Analysis Summary Table
        power_summary = power_df.groupby(['method', 'effect_size']).agg({
            'power': ['mean', 'std'],
            'sample_size': 'first'
        }).round(3)
        
        power_summary.columns = ['Mean_Power', 'Std_Power', 'Sample_Size']
        power_summary = power_summary.reset_index()
        
        # Add practical significance indicators
        power_summary['Power_Category'] = power_summary['Mean_Power'].apply(
            lambda x: 'Excellent (>=0.9)' if x >= 0.9 else 
                     'Good (>=0.8)' if x >= 0.8 else 
                     'Moderate (>=0.6)' if x >= 0.6 else 'Low (<0.6)'
        )
        
        if format_type == "latex":
            power_table = power_summary.to_latex(index=False, float_format='%.3f',
                                               caption="Statistical Power Analysis Summary",
                                               label="tab:power_analysis")
            power_path = os.path.join(output_dir, "power_analysis_table.tex")
            with open(power_path, 'w', encoding='utf-8') as f:
                f.write(power_table)
        else:
            power_path = os.path.join(output_dir, f"power_analysis_table.{format_type}")
            if format_type == "csv":
                power_summary.to_csv(power_path, index=False)
            elif format_type == "html":
                power_summary.to_html(power_path, index=False)
        
        table_paths['power_analysis'] = power_path
        
        # 2. Efficiency Analysis Summary Table
        if not efficiency_df.empty and not efficiency_df.dropna().empty:
            efficiency_summary = efficiency_df.groupby(['method']).agg({
                'execution_time': ['mean', 'std'],
                'memory_usage': ['mean', 'std'],
                'sample_size': ['min', 'max']
            }).round(3)
            
            efficiency_summary.columns = ['Mean_Time', 'Std_Time', 'Mean_Memory', 'Std_Memory', 'Min_N', 'Max_N']
            efficiency_summary = efficiency_summary.reset_index()
            
            if format_type == "latex":
                efficiency_table = efficiency_summary.to_latex(index=False, float_format='%.3f',
                                                             caption="Computational Efficiency Analysis Summary",
                                                             label="tab:efficiency_analysis")
                efficiency_path = os.path.join(output_dir, "efficiency_analysis_table.tex")
                with open(efficiency_path, 'w', encoding='utf-8') as f:
                    f.write(efficiency_table)
            else:
                efficiency_path = os.path.join(output_dir, f"efficiency_analysis_table.{format_type}")
                if format_type == "csv":
                    efficiency_summary.to_csv(efficiency_path, index=False)
                elif format_type == "html":
                    efficiency_summary.to_html(efficiency_path, index=False)
            
            table_paths['efficiency_analysis'] = efficiency_path
        
        return table_paths
    
    def create_publication_plots(
        self,
        results_df: pd.DataFrame,
        scalability_df: pd.DataFrame,
        output_dir: str = "plots",
        file_format: str = "pdf"
    ) -> Dict[str, str]:
        """Create all publication plots and save them.
        
        Args:
            results_df: DataFrame with simulation results
            scalability_df: DataFrame with scalability results
            output_dir: Directory to save plots
            file_format: File format for saving ('pdf', 'png', 'svg')
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        plot_paths = {}
        
        # Bias comparison plot
        fig, ax = self.bias_comparison_plot(results_df)
        bias_path = os.path.join(output_dir, f"bias_comparison.{file_format}")
        fig.savefig(bias_path, bbox_inches='tight', dpi=self.config['dpi'])
        plot_paths['bias_comparison'] = bias_path
        plt.close(fig)
        
        # RMSE comparison plot
        fig, ax = self.rmse_comparison_plot(results_df)
        rmse_path = os.path.join(output_dir, f"rmse_comparison.{file_format}")
        fig.savefig(rmse_path, bbox_inches='tight', dpi=self.config['dpi'])
        plot_paths['rmse_comparison'] = rmse_path
        plt.close(fig)
        
        # Scalability plot
        fig, ax = self.scalability_plot(scalability_df)
        scalability_path = os.path.join(output_dir, f"scalability_analysis.{file_format}")
        fig.savefig(scalability_path, bbox_inches='tight', dpi=self.config['dpi'])
        plot_paths['scalability_analysis'] = scalability_path
        plt.close(fig)
        
        return plot_paths

def create_publication_plots(
    results_df: pd.DataFrame,
    scalability_df: pd.DataFrame,
    output_dir: str = "plots",
    file_format: str = "pdf"
) -> Dict[str, str]:
    """Create all publication plots and save them.
    
    Args:
        results_df: DataFrame with simulation results
        scalability_df: DataFrame with scalability results
        output_dir: Directory to save plots
        file_format: File format for saving ('pdf', 'png', 'svg')
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    plotter = PublicationPlotter()
    saved_plots = {}
    
    # Bias comparison plot
    fig, ax = plotter.bias_comparison_plot(results_df)
    bias_path = os.path.join(output_dir, f"bias_comparison.{file_format}")
    fig.savefig(bias_path, dpi=300, bbox_inches='tight')
    saved_plots['bias_comparison'] = bias_path
    plt.close(fig)
    
    # RMSE comparison plot (if RMSE column exists)
    if 'rmse' in results_df.columns:
        fig, ax = plotter.rmse_comparison_plot(results_df)
        rmse_path = os.path.join(output_dir, f"rmse_comparison.{file_format}")
        fig.savefig(rmse_path, dpi=300, bbox_inches='tight')
        saved_plots['rmse_comparison'] = rmse_path
        plt.close(fig)
    
    # Scalability plot
    fig, ax = plotter.scalability_plot(scalability_df)
    scalability_path = os.path.join(output_dir, f"scalability_plot.{file_format}")
    fig.savefig(scalability_path, dpi=300, bbox_inches='tight')
    saved_plots['scalability'] = scalability_path
    plt.close(fig)
    
    # Statistical summary table
    analyzer = StatisticalAnalyzer()
    comparison_results = analyzer.comprehensive_method_comparison(results_df)
    fig, ax = plotter.create_summary_table_plot(comparison_results)
    summary_path = os.path.join(output_dir, f"statistical_summary.{file_format}")
    fig.savefig(summary_path, dpi=300, bbox_inches='tight')
    saved_plots['statistical_summary'] = summary_path
    plt.close(fig)
    
    return saved_plots
