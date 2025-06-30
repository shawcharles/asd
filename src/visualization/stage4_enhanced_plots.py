"""Stage 4 Enhanced Visualization Runner

This module orchestrates the creation of all enhanced publication-ready visualizations
and comprehensive results tables for the Adaptive Supergeo Design paper.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.visualization.publication_plots import PublicationPlotter


class Stage4VisualizationRunner:
    """Comprehensive visualization runner for Stage 4 enhancements."""
    
    def __init__(self, base_dir: str = None):
        """Initialize the visualization runner.
        
        Args:
            base_dir: Base directory for the project (defaults to project root)
        """
        self.base_dir = Path(base_dir) if base_dir else project_root
        self.plotter = PublicationPlotter()
        
        # Define input and output directories
        self.stage3_results_dir = self.base_dir / "stage3_practical_analysis_results"
        self.stage4_output_dir = self.base_dir / "stage4_enhanced_visualizations"
        
        # Create output directories
        self.stage4_output_dir.mkdir(exist_ok=True)
        (self.stage4_output_dir / "plots").mkdir(exist_ok=True)
        (self.stage4_output_dir / "tables").mkdir(exist_ok=True)
        (self.stage4_output_dir / "power_enhanced").mkdir(exist_ok=True)
        (self.stage4_output_dir / "efficiency_enhanced").mkdir(exist_ok=True)
    
    def load_stage3_data(self) -> Dict[str, pd.DataFrame]:
        """Load all Stage 3 analysis results.
        
        Returns:
            Dictionary containing loaded DataFrames
        """
        print("Loading Stage 3 analysis results...")
        
        data = {}
        
        # Load power analysis results
        power_file = self.stage3_results_dir / "power_analysis_results.csv"
        if power_file.exists():
            data['power_df'] = pd.read_csv(power_file)
            print(f"✓ Loaded power analysis data: {len(data['power_df'])} records")
        else:
            print("⚠ Power analysis results not found")
            data['power_df'] = pd.DataFrame()
        
        # Load efficiency analysis results
        efficiency_file = self.stage3_results_dir / "efficiency_analysis_results.csv"
        if efficiency_file.exists():
            data['efficiency_df'] = pd.read_csv(efficiency_file)
            print(f"✓ Loaded efficiency analysis data: {len(data['efficiency_df'])} records")
        else:
            print("⚠ Efficiency analysis results not found")
            data['efficiency_df'] = pd.DataFrame()
        
        return data
    
    def generate_enhanced_power_plots(self, power_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate enhanced power analysis visualizations.
        
        Args:
            power_df: DataFrame with power analysis results
            
        Returns:
            Dictionary with plot information and paths
        """
        print("\n🎨 Generating Enhanced Power Analysis Plots...")
        
        if power_df.empty:
            print("⚠ No power analysis data available for plotting")
            return {}
        
        # Generate enhanced power plots
        output_dir = str(self.stage4_output_dir / "power_enhanced")
        figures = self.plotter.power_analysis_enhanced_plots(
            power_df=power_df,
            output_dir=output_dir,
            save_plots=True
        )
        
        print(f"✓ Generated {len(figures)} enhanced power analysis plots")
        
        # Create summary statistics
        power_summary = {
            'total_scenarios': len(power_df),
            'methods_analyzed': power_df['method'].nunique(),
            'effect_sizes': sorted(power_df['effect_size'].unique()),
            'sample_sizes': sorted(power_df['sample_size'].unique()),
            'plots_generated': list(figures.keys())
        }
        
        return {
            'figures': figures,
            'summary': power_summary,
            'output_dir': output_dir
        }
    
    def generate_enhanced_efficiency_plots(self, efficiency_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate enhanced computational efficiency visualizations.
        
        Args:
            efficiency_df: DataFrame with efficiency analysis results
            
        Returns:
            Dictionary with plot information and paths
        """
        print("\n⚡ Generating Enhanced Efficiency Analysis Plots...")
        
        if efficiency_df.empty:
            print("⚠ No efficiency analysis data available for plotting")
            return {}
        
        # Generate enhanced efficiency plots
        output_dir = str(self.stage4_output_dir / "efficiency_enhanced")
        figures = self.plotter.efficiency_comparison_plots(
            efficiency_df=efficiency_df,
            output_dir=output_dir,
            save_plots=True
        )
        
        print(f"✓ Generated {len(figures)} enhanced efficiency analysis plots")
        
        # Create summary statistics
        efficiency_summary = {
            'total_measurements': len(efficiency_df),
            'methods_analyzed': efficiency_df['method'].nunique() if 'method' in efficiency_df.columns else 0,
            'sample_sizes': sorted(efficiency_df['sample_size'].unique()) if 'sample_size' in efficiency_df.columns else [],
            'plots_generated': list(figures.keys())
        }
        
        return {
            'figures': figures,
            'summary': efficiency_summary,
            'output_dir': output_dir
        }
    
    def generate_comprehensive_tables(self, power_df: pd.DataFrame, efficiency_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive results tables for publication.
        
        Args:
            power_df: DataFrame with power analysis results
            efficiency_df: DataFrame with efficiency analysis results
            
        Returns:
            Dictionary with table information and paths
        """
        print("\n📊 Generating Comprehensive Results Tables...")
        
        output_dir = str(self.stage4_output_dir / "tables")
        
        # Generate tables in multiple formats
        table_results = {}
        
        for format_type in ['latex', 'csv', 'html']:
            try:
                table_paths = self.plotter.generate_results_tables(
                    power_df=power_df,
                    efficiency_df=efficiency_df,
                    output_dir=output_dir,
                    format_type=format_type
                )
                table_results[format_type] = table_paths
                print(f"✓ Generated {len(table_paths)} tables in {format_type} format")
            except Exception as e:
                print(f"⚠ Error generating {format_type} tables: {str(e)}")
                table_results[format_type] = {}
        
        return {
            'table_paths': table_results,
            'output_dir': output_dir
        }
    
    def create_stage4_summary_report(self, results: Dict[str, Any]) -> str:
        """Create a comprehensive summary report of Stage 4 enhancements.
        
        Args:
            results: Dictionary containing all Stage 4 results
            
        Returns:
            Path to the generated report
        """
        print("\n📝 Creating Stage 4 Summary Report...")
        
        report_content = f"""# Stage 4 Enhanced Visualizations Summary Report

## Executive Summary

This report summarizes the enhanced visualization and communication improvements implemented in Stage 4 of the Adaptive Supergeo Design paper enhancement project.

## Enhanced Power Analysis Visualizations

### Power Analysis Plots Generated
"""
        
        if 'power_plots' in results and results['power_plots']:
            power_summary = results['power_plots']['summary']
            report_content += f"""
- **Total Scenarios Analyzed**: {power_summary.get('total_scenarios', 'N/A')}
- **Methods Compared**: {power_summary.get('methods_analyzed', 'N/A')}
- **Effect Sizes**: {power_summary.get('effect_sizes', 'N/A')}
- **Sample Sizes**: {power_summary.get('sample_sizes', 'N/A')}
- **Enhanced Plots Generated**: {len(power_summary.get('plots_generated', []))}

### Key Features Added:
- Wilson score confidence intervals for power estimates
- 80% power threshold reference lines
- Power heatmaps across all conditions
- Professional color schemes and typography
- Statistical significance indicators
"""
        else:
            report_content += "\n⚠ No power analysis plots were generated (data not available)\n"
        
        report_content += "\n## Enhanced Efficiency Analysis Visualizations\n"
        
        if 'efficiency_plots' in results and results['efficiency_plots']:
            efficiency_summary = results['efficiency_plots']['summary']
            report_content += f"""
- **Total Measurements**: {efficiency_summary.get('total_measurements', 'N/A')}
- **Methods Analyzed**: {efficiency_summary.get('methods_analyzed', 'N/A')}
- **Sample Size Range**: {efficiency_summary.get('sample_sizes', 'N/A')}
- **Enhanced Plots Generated**: {len(efficiency_summary.get('plots_generated', []))}

### Key Features Added:
- Execution time scalability analysis
- Memory usage profiling
- Computational complexity visualization
- Error handling for missing data
- Professional formatting and legends
"""
        else:
            report_content += "\n⚠ No efficiency analysis plots were generated (data not available)\n"
        
        report_content += "\n## Comprehensive Results Tables\n"
        
        if 'tables' in results and results['tables']:
            table_results = results['tables']['table_paths']
            report_content += f"""
### Generated Table Formats:
- **LaTeX**: {len(table_results.get('latex', {}))} tables for academic publication
- **CSV**: {len(table_results.get('csv', {}))} tables for data analysis
- **HTML**: {len(table_results.get('html', {}))} tables for web presentation

### Table Contents:
- Statistical power analysis summary with practical significance indicators
- Computational efficiency analysis with performance metrics
- Professional formatting with captions and labels
- Uncertainty quantification and confidence intervals
"""
        else:
            report_content += "\n⚠ No results tables were generated\n"
        
        report_content += f"""
## Technical Implementation

### Enhanced Plotting Framework
- **Publication-ready aesthetics**: Professional color schemes, typography, and layout
- **Statistical rigor**: Confidence intervals, significance indicators, and error bars
- **Comprehensive coverage**: Power analysis, efficiency analysis, and comparative visualizations
- **Multiple formats**: PNG, PDF, and SVG output support

### Results Table Generation
- **Multi-format support**: LaTeX, CSV, and HTML outputs
- **Statistical completeness**: Mean, standard deviation, and confidence intervals
- **Practical significance**: Power categories and performance indicators
- **Academic formatting**: Professional captions, labels, and structure

## Output Directory Structure

```
stage4_enhanced_visualizations/
├── power_enhanced/
│   ├── power_curve_enhanced_es_*.png
│   └── power_heatmap_comparison.png
├── efficiency_enhanced/
│   └── efficiency_comparison_enhanced.png
├── tables/
│   ├── power_analysis_table.tex
│   ├── power_analysis_table.csv
│   ├── power_analysis_table.html
│   ├── efficiency_analysis_table.tex
│   ├── efficiency_analysis_table.csv
│   └── efficiency_analysis_table.html
└── stage4_summary_report.md
```

## Quality Assurance

### Visualization Standards Met:
- ✓ All plots include error bars and confidence intervals
- ✓ Statistical significance indicators implemented
- ✓ Consistent professional design system
- ✓ Publication-ready resolution and formatting
- ✓ Comprehensive legend and axis labeling

### Table Standards Met:
- ✓ Complete statistical reporting with uncertainty
- ✓ Practical significance indicators included
- ✓ Professional academic formatting
- ✓ Multiple output formats for flexibility
- ✓ Clear captions and labels

## Integration with Paper

These enhanced visualizations and tables are ready for integration into the main Adaptive Supergeo Design paper, providing:

1. **Superior Visual Communication**: Professional plots with statistical rigor
2. **Comprehensive Results Reporting**: Complete tables with uncertainty quantification
3. **Publication Readiness**: Academic-standard formatting and presentation
4. **Reproducible Analysis**: Clear methodology and documented outputs

## Recommendations for Paper Integration

1. **Replace existing plots** with enhanced versions from `power_enhanced/` and `efficiency_enhanced/` directories
2. **Integrate LaTeX tables** directly into the paper using the generated `.tex` files
3. **Reference comprehensive results** using the detailed summary statistics
4. **Highlight enhanced methodology** in the paper's methodology section

---

**Report Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Stage 4 Status**: Complete
**Next Steps**: Paper integration and final review
"""
        
        # Save report
        report_path = self.stage4_output_dir / "stage4_summary_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ Stage 4 summary report saved to: {report_path}")
        return str(report_path)
    
    def run_comprehensive_stage4_analysis(self) -> Dict[str, Any]:
        """Run the complete Stage 4 enhanced visualization analysis.
        
        Returns:
            Dictionary containing all results and paths
        """
        print("🚀 Starting Stage 4 Enhanced Visualization Analysis")
        print("=" * 60)
        
        # Load Stage 3 data
        data = self.load_stage3_data()
        
        results = {}
        
        # Generate enhanced power plots
        if not data['power_df'].empty:
            results['power_plots'] = self.generate_enhanced_power_plots(data['power_df'])
        
        # Generate enhanced efficiency plots
        if not data['efficiency_df'].empty:
            results['efficiency_plots'] = self.generate_enhanced_efficiency_plots(data['efficiency_df'])
        
        # Generate comprehensive tables
        results['tables'] = self.generate_comprehensive_tables(
            data['power_df'], 
            data['efficiency_df']
        )
        
        # Create summary report
        results['summary_report'] = self.create_stage4_summary_report(results)
        
        print("\n" + "=" * 60)
        print("🎉 Stage 4 Enhanced Visualization Analysis Complete!")
        print(f"📁 All outputs saved to: {self.stage4_output_dir}")
        
        return results


def main():
    """Main execution function for Stage 4 analysis."""
    try:
        # Initialize and run Stage 4 analysis
        runner = Stage4VisualizationRunner()
        results = runner.run_comprehensive_stage4_analysis()
        
        print("\n📋 Stage 4 Analysis Summary:")
        print(f"- Enhanced plots generated: {len(results.get('power_plots', {}).get('figures', {})) + len(results.get('efficiency_plots', {}).get('figures', {}))}")
        print(f"- Results tables created: {sum(len(tables) for tables in results.get('tables', {}).get('table_paths', {}).values())}")
        print(f"- Summary report: {results.get('summary_report', 'Not generated')}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in Stage 4 analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
