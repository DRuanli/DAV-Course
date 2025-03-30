"""
Statistical Visualizations Module for Heart Disease Dataset

This module implements specialized statistical visualizations for the UCI
Heart Disease dataset, focusing on hypothesis test results, probability
distributions, and statistical findings.

Author: Team Member 2
Date: March 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union
import logging
from matplotlib.gridspec import GridSpec
from scipy import stats
import matplotlib.patches as mpatches

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define custom color palettes
HYPOTHESIS_PALETTE = ['#4878D0', '#EE854A']  # Blue/Orange
SIGNIFICANCE_COLORS = {'significant': '#EE854A', 'non-significant': '#4878D0'}


def setup_visualization_env():
    """Set up the visualization environment with customized settings for statistical plots."""
    # Set seaborn style for statistical visualization
    sns.set_style("whitegrid")

    # Statistical-themed color palette
    colors = ["#2C7FB8", "#7FCDBB", "#D95F0E", "#FEC44F", "#FFFFCC", "#FC8D59"]
    sns.set_palette(sns.color_palette(colors))

    # Set matplotlib parameters for consistent statistical visualizations
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    # Title style that looks more like statistical publications
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['figure.titleweight'] = 'bold'


def create_hypothesis_test_visualizations(df: pd.DataFrame,
                                          test_results: Dict,
                                          output_dir: str) -> Dict[str, str]:
    """
    Create visualizations for hypothesis test results.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    test_results : Dict
        Dictionary containing hypothesis test results
    output_dir : str
        Directory to save the visualizations

    Returns
    -------
    Dict[str, str]
        Dictionary mapping test names to output file paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_files = {}

    # Check if we have continuous variable tests
    if 'continuous_variable_tests' in test_results:
        continuous_tests = test_results['continuous_variable_tests']

        for test_name, result in continuous_tests.items():
            if 'error' in result:
                continue

            # Extract variable name from test name
            var_name = test_name.split('_')[0]

            # Create figure with two panels for t-test results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

            # Extract data for the groups
            group_keys = sorted(result.get('means', {}).keys())
            means = [result['means'].get(k, 0) for k in group_keys]

            # Define better labels
            if len(group_keys) == 2 and group_keys == ['0', '1']:
                labels = ['No Disease', 'Disease']
            else:
                labels = group_keys

            # Left panel: Mean comparison with confidence intervals
            error_bars = None
            if 'confidence_interval' in result:
                ci = result['confidence_interval']
                mean_diff = result.get('mean_difference', means[1] - means[0])
                error_bars = [ci['upper'] - mean_diff, mean_diff - ci['lower']]

            # Create bar chart
            bars = ax1.bar(labels, means, color=HYPOTHESIS_PALETTE, yerr=error_bars)

            # Add p-value and significance annotation
            p_value = result.get('p_value', 1.0)
            sig_text = "* Significant difference" if p_value < 0.05 else "Not significant"
            ax1.text(0.5, 0.95, f"p-value: {p_value:.4f}\n{sig_text}",
                     transform=ax1.transAxes, ha='center', va='top',
                     bbox=dict(facecolor='white', alpha=0.8))

            # Add mean values on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                         f'{height:.2f}', ha='center', va='bottom')

            # Right panel: Distribution comparison
            # Extract actual data from the dataset
            if var_name in df.columns and 'target_binary' in df.columns:
                # Create violin plot
                sns.violinplot(x='target_binary', y=var_name, data=df,
                               palette=HYPOTHESIS_PALETTE, ax=ax2)

                # Add individual points
                sns.stripplot(x='target_binary', y=var_name, data=df,
                              color='black', alpha=0.3, jitter=True, size=3, ax=ax2)

                # Set better x-tick labels
                ax2.set_xticklabels(labels)

                # Add effect size annotation
                if 'effect_size' in result:
                    effect_size = result['effect_size']
                    ax2.text(0.5, 0.95,
                             f"Effect size ({effect_size['method']}): {effect_size['value']:.3f}\n" +
                             f"Interpretation: {effect_size['interpretation']}",
                             transform=ax2.transAxes, ha='center', va='top',
                             bbox=dict(facecolor='white', alpha=0.8))

            # Set titles and labels
            ax1.set_title(f'Mean {var_name} by Group')
            ax1.set_ylabel(var_name)
            ax2.set_title(f'Distribution of {var_name} by Group')
            ax2.set_ylabel(var_name)
            ax2.set_xlabel('Group')

            # Add overall title
            test_used = result.get('test_used', 'Statistical Test')
            fig.suptitle(f"{var_name} Comparison: {test_used}", fontsize=16)

            # Add conclusion text
            if 'conclusion' in result:
                conclusion = result['conclusion']
                fig.text(0.5, 0.01, conclusion, ha='center', fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))

            plt.tight_layout()
            plt.subplots_adjust(top=0.9, bottom=0.15)

            # Save the figure
            output_file = f"{output_dir}/{test_name}_visualization.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            output_files[test_name] = output_file

    # Check if we have categorical association tests
    if 'categorical_association_tests' in test_results:
        categorical_tests = test_results['categorical_association_tests']

        for test_name, result in categorical_tests.items():
            if 'error' in result:
                continue

            # Extract variable name from test name
            var_name = test_name.split('_')[0]

            # Create figure with two panels for chi-square test results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

            # Extract contingency table data
            if 'contingency_table' in result:
                contingency = result['contingency_table']

                # Check if we have a valid contingency table
                if not contingency:
                    continue

                # Convert to dataframe if it's a dict
                if isinstance(contingency, dict):
                    contingency_df = pd.DataFrame(contingency)
                else:
                    contingency_df = contingency

                # Create a heatmap of the contingency table
                sns.heatmap(contingency_df, annot=True, fmt='d', cmap='Blues', ax=ax1)
                ax1.set_title(f'Contingency Table for {var_name} vs. Disease')

                # If we have percentage data, create a grouped bar chart
                if 'percentage_table' in result:
                    percentages = result['percentage_table']

                    # Convert to dataframe if it's a dict
                    if isinstance(percentages, dict):
                        pct_df = pd.DataFrame(percentages)
                    else:
                        pct_df = percentages

                    # Reshape for plotting
                    pct_df_plot = pct_df.reset_index().melt(id_vars='index',
                                                            var_name='target',
                                                            value_name='percentage')
                    pct_df_plot.columns = [var_name, 'target', 'percentage']

                    # Create grouped bar chart
                    sns.barplot(x=var_name, y='percentage', hue='target',
                                data=pct_df_plot, palette=HYPOTHESIS_PALETTE, ax=ax2)

                    # Improve labels
                    ax2.set_title(f'Percentage of Disease by {var_name}')
                    ax2.set_ylabel('Percentage (%)')
                    ax2.legend(title='Disease Status', labels=['No Disease', 'Disease'])

                    # Add chi-square test result
                    chi2 = result.get('statistic', 0)
                    p_value = result.get('p_value', 1.0)

                    # Add effect size if available
                    if 'effect_size' in result:
                        effect = result['effect_size']
                        effect_text = f"\nEffect size ({effect['method']}): {effect['value']:.3f} ({effect['interpretation']})"
                    else:
                        effect_text = ""

                    ax2.text(0.5, 0.95,
                             f"Chi-square: {chi2:.2f}, p-value: {p_value:.4f}" +
                             f"\nSignificant: {'Yes' if p_value < 0.05 else 'No'}" +
                             effect_text,
                             transform=ax2.transAxes, ha='center', va='top',
                             bbox=dict(facecolor='white', alpha=0.8))

            # Add overall title
            fig.suptitle(f"Association Between {var_name} and Disease", fontsize=16)

            # Add conclusion text
            if 'conclusion' in result:
                conclusion = result['conclusion']
                fig.text(0.5, 0.01, conclusion, ha='center', fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))

            plt.tight_layout()
            plt.subplots_adjust(top=0.9, bottom=0.15)

            # Save the figure
            output_file = f"{output_dir}/{test_name}_visualization.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            output_files[test_name] = output_file

    return output_files


def create_distribution_analysis_visualizations(df: pd.DataFrame,
                                                analysis_results: Dict,
                                                output_dir: str) -> Dict[str, str]:
    """
    Create visualizations for distribution analysis results.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    analysis_results : Dict
        Dictionary containing distribution analysis results
    output_dir : str
        Directory to save the visualizations

    Returns
    -------
    Dict[str, str]
        Dictionary mapping variable names to output file paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    from scipy import stats

    output_files = {}

    # Check if we have variable distributions
    if 'variable_distributions' in analysis_results:
        var_distributions = analysis_results['variable_distributions']

        for var_name, results in var_distributions.items():
            # Create a figure with multiple panels for the distribution analysis
            fig = plt.figure(figsize=(15, 12))
            gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

            # Panel 1: Distribution plot
            ax1 = fig.add_subplot(gs[0, 0])

            # Get the distribution data
            if var_name in df.columns:
                data = df[var_name].dropna()

                # Plot histogram with KDE
                sns.histplot(data, kde=True, ax=ax1, color='#4878D0')

                # Add normal curve
                if results.get('is_normal', False):
                    x = np.linspace(data.min(), data.max(), 100)
                    mu, sigma = results.get('mean', data.mean()), results.get('std', data.std())
                    y = stats.norm.pdf(x, mu, sigma) * len(data) * (data.max() - data.min()) / 30
                    ax1.plot(x, y, 'r--', linewidth=2, label=f'Normal: μ={mu:.2f}, σ={sigma:.2f}')

                # Add mean and median lines
                ax1.axvline(results.get('mean', data.mean()), color='red', linestyle='-',
                            linewidth=2, label=f"Mean: {results.get('mean', data.mean()):.2f}")
                ax1.axvline(results.get('median', data.median()), color='green', linestyle='--',
                            linewidth=2, label=f"Median: {results.get('median', data.median()):.2f}")

                ax1.set_title(f'Distribution of {var_name}')
                ax1.legend()

            # Panel 2: Q-Q Plot
            ax2 = fig.add_subplot(gs[0, 1])

            # Create Q-Q plot if provided in results
            if 'qq_plot' in results:
                qq_path = results['qq_plot']

                # If it's a path to an image, we'll recreate the Q-Q plot
                if var_name in df.columns:
                    data = df[var_name].dropna()
                    # Create Q-Q plot
                    stats.probplot(data, dist="norm", plot=ax2)
                    ax2.set_title(f'Q-Q Plot for {var_name}')

            # Panel 3: Normality Test Results
            ax3 = fig.add_subplot(gs[1, 0])

            # Create a table with normality test results
            if 'normality_tests' in results:
                tests = results['normality_tests']

                # Create data for the table
                test_names = []
                statistics = []
                p_values = []
                conclusions = []

                for test_name, test_result in tests.items():
                    if 'error' in test_result:
                        continue

                    test_names.append(test_name.capitalize())
                    statistics.append(f"{test_result.get('statistic', 'N/A'):.4f}")
                    p_values.append(f"{test_result.get('p_value', 'N/A'):.4f}")

                    if test_result.get('normal', False):
                        conclusions.append('Normal')
                    else:
                        conclusions.append('Non-normal')

                # Create table
                table_data = [test_names, statistics, p_values, conclusions]
                table = ax3.table(cellText=table_data,
                                  rowLabels=['Test', 'Statistic', 'p-value', 'Conclusion'],
                                  loc='center', cellLoc='center', colWidths=[0.25] * len(test_names))
                table.auto_set_font_size(False)
                table.set_fontsize(11)
                table.scale(1, 1.5)

                # Hide axis
                ax3.axis('off')
                ax3.set_title('Normality Tests', pad=20)

            # Panel 4: Distribution Information and Interpretation
            ax4 = fig.add_subplot(gs[1, 1])

            # Create a text box with distribution information
            stats_text = (
                f"Sample Size: {results.get('count', 'N/A')}\n"
                f"Mean: {results.get('mean', 'N/A'):.2f}\n"
                f"Median: {results.get('median', 'N/A'):.2f}\n"
                f"Std Dev: {results.get('std', 'N/A'):.2f}\n"
                f"Range: {results.get('range', 'N/A'):.2f}\n"
                f"IQR: {results.get('iqr', 'N/A'):.2f}\n"
                f"Skewness: {results.get('skewness', 'N/A'):.2f}\n"
                f"Kurtosis: {results.get('kurtosis', 'N/A'):.2f}\n"
                f"Distribution Type: {results.get('distribution_type', 'N/A')}\n\n"
                f"Medical Interpretation:\n{results.get('medical_interpretation', 'N/A')}"
            )

            ax4.text(0.5, 0.5, stats_text, transform=ax4.transAxes,
                     ha='center', va='center', fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            ax4.axis('off')
            ax4.set_title('Distribution Statistics', pad=20)

            # Add overall title
            fig.suptitle(f'Distribution Analysis of {var_name}', fontsize=18)

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)

            # Save the figure
            output_file = f"{output_dir}/{var_name}_distribution_analysis.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            output_files[var_name] = output_file

            # Create group comparison plots if available
            if 'segment_analyses' in analysis_results and var_name in analysis_results['segment_analyses']:
                segment_results = analysis_results['segment_analyses'][var_name]

                for segment_var, segment_analysis in segment_results.items():
                    # Create a figure for segment analysis
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

                    # Left panel: Distribution by segment
                    if 'segment_plot' in segment_analysis and var_name in df.columns and segment_var in df.columns:
                        # Create violin plot
                        for segment in df[segment_var].unique():
                            segment_data = df[df[segment_var] == segment][var_name].dropna()
                            if len(segment_data) > 0:
                                sns.kdeplot(segment_data, ax=ax1, label=str(segment), fill=True, alpha=0.3)

                        ax1.set_title(f'Distribution of {var_name} by {segment_var}')
                        ax1.set_xlabel(var_name)
                        ax1.legend(title=segment_var)

                    # Right panel: Segment comparison
                    if 'segment_stats' in segment_analysis:
                        segment_stats = segment_analysis['segment_stats']

                        # Extract means for each segment
                        segments = []
                        means = []

                        for segment, stats in segment_stats.items():
                            segments.append(str(segment))
                            means.append(stats.get('mean', 0))

                        # Create bar chart
                        colors = sns.color_palette("Set2", len(segments))
                        bars = ax2.bar(segments, means, color=colors)

                        # Add mean values on top of bars
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                                     f'{height:.2f}', ha='center', va='bottom')

                        ax2.set_title(f'Mean {var_name} by {segment_var}')
                        ax2.set_xlabel(segment_var)
                        ax2.set_ylabel(f'Mean {var_name}')

                        # Add statistical test result if available
                        if 'difference_test' in segment_analysis:
                            test = segment_analysis['difference_test']

                            test_name = test.get('test', 'Statistical test')
                            stat = test.get('statistic', 0)
                            p_val = test.get('p_value', 1.0)

                            sig_text = "* Significant difference" if test.get('significant',
                                                                              False) else "Not significant"

                            ax2.text(0.5, 0.95,
                                     f"{test_name}: {stat:.2f}, p-value: {p_val:.4f}\n{sig_text}",
                                     transform=ax2.transAxes, ha='center', va='top',
                                     bbox=dict(facecolor='white', alpha=0.8))

                    # Add conclusion text
                    if 'medical_interpretation' in segment_analysis:
                        interp = segment_analysis['medical_interpretation']
                        fig.text(0.5, 0.01, interp, ha='center', fontsize=12,
                                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))

                    plt.tight_layout()
                    plt.subplots_adjust(bottom=0.15)

                    # Save the figure
                    output_file = f"{output_dir}/{var_name}_by_{segment_var}_analysis.png"
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close(fig)

                    output_files[f"{var_name}_by_{segment_var}"] = output_file

    return output_files


def create_correlation_visualizations(df: pd.DataFrame,
                                      analysis_results: Dict,
                                      output_dir: str) -> Dict[str, str]:
    """
    Create visualizations for correlation analysis results.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    analysis_results : Dict
        Dictionary containing correlation analysis results
    output_dir : str
        Directory to save the visualizations

    Returns
    -------
    Dict[str, str]
        Dictionary mapping visualization types to output file paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_files = {}

    # Create key correlations with target visualization
    if 'key_correlations' in analysis_results:
        key_correlations = analysis_results['key_correlations']

        # Create a figure for top correlations
        fig, ax = plt.subplots(figsize=(12, 8))

        # Extract data for top 10 correlations
        top_correlations = key_correlations[:10]

        vars = [f"{c['var1']} - {c['var2']}" for c in top_correlations]
        correlation_values = [c['value'] for c in top_correlations]
        significances = [c.get('significant', False) for c in top_correlations]

        # Define colors based on significance
        colors = [SIGNIFICANCE_COLORS['significant'] if sig else SIGNIFICANCE_COLORS['non-significant']
                  for sig in significances]

        # Create horizontal bar chart
        y_pos = np.arange(len(vars))
        bars = ax.barh(y_pos, correlation_values, color=colors)

        # Set labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(vars)
        ax.set_xlabel('Correlation Coefficient')
        ax.set_title('Top Correlations in Heart Disease Dataset')

        # Add a vertical line at 0
        ax.axvline(x=0, color='gray', linestyle='--')

        # Add correlation values
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x = width + 0.02 if width >= 0 else width - 0.08
            ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                    f'{width:.2f}', va='center')

        # Add legend for significance
        significant_patch = mpatches.Patch(color=SIGNIFICANCE_COLORS['significant'],
                                           label='Significant (p < 0.05)')
        non_significant_patch = mpatches.Patch(color=SIGNIFICANCE_COLORS['non-significant'],
                                               label='Not Significant')
        ax.legend(handles=[significant_patch, non_significant_patch], loc='lower right')

        plt.tight_layout()

        # Save the figure
        output_file = f"{output_dir}/top_correlations.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        output_files['top_correlations'] = output_file

    # Create partial correlation network visualization
    if 'partial_correlation_insights' in analysis_results and len(analysis_results['partial_correlation_insights']) > 0:
        insights = analysis_results['partial_correlation_insights']

        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create text box with insights
        text = "\n".join([f"• {insight}" for insight in insights])
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        ax.text(0.5, 0.5, text, transform=ax.transAxes,
                fontsize=12, verticalalignment='center',
                horizontalalignment='center', bbox=props)

        ax.set_title('Partial Correlation Insights')
        ax.axis('off')

        plt.tight_layout()

        # Save the figure
        output_file = f"{output_dir}/partial_correlation_insights.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        output_files['partial_correlation_insights'] = output_file

    # Create correlation difference visualization
    if 'correlation_differences' in analysis_results:
        corr_diffs = analysis_results['correlation_differences']

        if corr_diffs and len(corr_diffs) > 0:
            # Create a figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Extract data for top differences
            top_diffs = corr_diffs[:min(len(corr_diffs), 10)]

            var_pairs = [f"{d['variable1']} - {d['variable2']}" for d in top_diffs]
            pearson_values = [d['pearson'] for d in top_diffs]
            spearman_values = [d['spearman'] for d in top_diffs]

            # Create a dataframe for plotting
            plot_data = pd.DataFrame({
                'Variable Pair': var_pairs,
                'Pearson': pearson_values,
                'Spearman': spearman_values
            })

            # Reshape for grouped bar chart
            plot_data_long = pd.melt(plot_data, id_vars=['Variable Pair'],
                                     value_vars=['Pearson', 'Spearman'],
                                     var_name='Correlation Type', value_name='Correlation')

            # Create grouped bar chart
            sns.barplot(x='Variable Pair', y='Correlation', hue='Correlation Type',
                        data=plot_data_long, ax=ax)

            # Rotate x-tick labels
            plt.xticks(rotation=45, ha='right')

            ax.set_title('Differences Between Pearson and Spearman Correlations')
            ax.set_xlabel('')

            plt.tight_layout()

            # Save the figure
            output_file = f"{output_dir}/correlation_differences.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            output_files['correlation_differences'] = output_file

    # Create clinical interpretations visualization
    if 'clinical_interpretations' in analysis_results:
        interpretations = analysis_results['clinical_interpretations']

        if interpretations and len(interpretations) > 0:
            # Create a figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Create text box with interpretations
            text = "\n\n".join([f"{i + 1}. {interp}" for i, interp in enumerate(interpretations)])
            props = dict(boxstyle='round', facecolor='white', alpha=0.9)
            ax.text(0.5, 0.5, text, transform=ax.transAxes,
                    fontsize=12, verticalalignment='center',
                    horizontalalignment='center', bbox=props, wrap=True)

            ax.set_title('Clinical Interpretations Based on Correlation Analysis')
            ax.axis('off')

            plt.tight_layout()

            # Save the figure
            output_file = f"{output_dir}/clinical_interpretations.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            output_files['clinical_interpretations'] = output_file

    return output_files


def create_statistical_visualizations(df: pd.DataFrame,
                                      analysis_results: Dict,
                                      output_dir: str = 'results/phase4/statistical_viz') -> Dict[str, Dict[str, str]]:
    """
    Create all statistical visualizations for the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset for visualization
    analysis_results : Dict
        Dictionary containing analysis results from previous phases
    output_dir : str
        Directory to save figures (default: 'results/phase4/statistical_viz')

    Returns
    -------
    Dict[str, Dict[str, str]]
        Dictionary mapping visualization types to their output files
    """
    # Ensure visualization environment is set up
    setup_visualization_env()

    # Create output directories
    hypothesis_dir = os.path.join(output_dir, 'hypothesis_tests')
    distribution_dir = os.path.join(output_dir, 'distributions')
    correlation_dir = os.path.join(output_dir, 'correlations')

    os.makedirs(hypothesis_dir, exist_ok=True)
    os.makedirs(distribution_dir, exist_ok=True)
    os.makedirs(correlation_dir, exist_ok=True)

    visualizations = {}

    # Create hypothesis test visualizations
    if 'hypothesis_testing' in analysis_results:
        print("\nCreating hypothesis test visualizations...")
        hypothesis_viz = create_hypothesis_test_visualizations(
            df,
            analysis_results['hypothesis_testing'],
            hypothesis_dir
        )
        visualizations['hypothesis_tests'] = hypothesis_viz

    # Create distribution analysis visualizations
    if 'distribution_analysis' in analysis_results:
        print("\nCreating distribution analysis visualizations...")
        distribution_viz = create_distribution_analysis_visualizations(
            df,
            analysis_results['distribution_analysis'],
            distribution_dir
        )
        visualizations['distributions'] = distribution_viz

    # Create correlation analysis visualizations
    if 'correlation_analysis' in analysis_results:
        print("\nCreating correlation analysis visualizations...")
        correlation_viz = create_correlation_visualizations(
            df,
            analysis_results['correlation_analysis'],
            correlation_dir
        )
        visualizations['correlations'] = correlation_viz

    print(f"\nAll statistical visualizations created and saved to {output_dir}")

    return visualizations


if __name__ == "__main__":
    # Test the module if run directly
    print("Statistical Visualizations Module for the Heart Disease dataset.")
    print("This module provides statistical visualization functions.")
    print("Import and use in another script or notebook.")