"""
Probability Distribution Analysis Module for Heart Disease Dataset

This module implements formal distribution analysis for key variables in the
UCI Heart Disease dataset, including normality tests and visualization.

Author: Team Member 1
Date: March 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Union
from statsmodels.graphics.gofplots import qqplot
import logging

# Set up the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def test_normality(data: pd.Series) -> Dict:
    """
    Perform normality tests on a data series.

    Parameters
    ----------
    data : pd.Series
        Data series to test for normality

    Returns
    -------
    Dict
        Dictionary containing test results
    """
    results = {}

    # Shapiro-Wilk test
    # Most powerful for small to medium sample sizes
    try:
        shapiro_stat, shapiro_p = stats.shapiro(data.dropna())
        results['shapiro'] = {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'normal': shapiro_p > 0.05
        }
    except Exception as e:
        logging.warning(f"Shapiro-Wilk test failed: {e}")
        results['shapiro'] = {
            'error': str(e)
        }

    # Kolmogorov-Smirnov test
    # Compares data with normal distribution
    try:
        ks_stat, ks_p = stats.kstest(stats.zscore(data.dropna()), 'norm')
        results['ks'] = {
            'statistic': ks_stat,
            'p_value': ks_p,
            'normal': ks_p > 0.05
        }
    except Exception as e:
        logging.warning(f"Kolmogorov-Smirnov test failed: {e}")
        results['ks'] = {
            'error': str(e)
        }

    # Anderson-Darling test
    # More sensitive to deviations in the tails of the distribution
    try:
        ad_result = stats.anderson(data.dropna(), 'norm')
        results['anderson'] = {
            'statistic': ad_result.statistic,
            'critical_values': ad_result.critical_values.tolist(),
            'significance_levels': ad_result.significance_level.tolist(),
            'normal': ad_result.statistic < ad_result.critical_values[2]  # Using 5% significance level
        }
    except Exception as e:
        logging.warning(f"Anderson-Darling test failed: {e}")
        results['anderson'] = {
            'error': str(e)
        }

    return results


def create_qq_plot(data: pd.Series, variable_name: str, output_dir: str = 'results/phase3/figures') -> str:
    """
    Create a QQ plot for visual distribution assessment.

    Parameters
    ----------
    data : pd.Series
        Data series to plot
    variable_name : str
        Name of the variable for the plot title and filename
    output_dir : str
        Directory to save the plot

    Returns
    -------
    str
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create the QQ plot
    fig, ax = plt.subplots(figsize=(10, 8))
    qqplot(data.dropna(), line='s', ax=ax)

    # Add title and labels
    ax.set_title(f'Q-Q Plot for {variable_name}', fontsize=16)
    ax.set_xlabel('Theoretical Quantiles', fontsize=14)
    ax.set_ylabel('Sample Quantiles', fontsize=14)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Save the plot
    output_file = os.path.join(output_dir, f'{variable_name.replace(" ", "_").lower()}_qq_plot.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return output_file


def create_distribution_plot(data: pd.Series, variable_name: str,
                             output_dir: str = 'results/phase3/figures') -> str:
    """
    Create a distribution plot with histogram, KDE, and normal distribution overlay.

    Parameters
    ----------
    data : pd.Series
        Data series to plot
    variable_name : str
        Name of the variable for the plot title and filename
    output_dir : str
        Directory to save the plot

    Returns
    -------
    str
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot histogram with KDE
    sns.histplot(data.dropna(), kde=True, ax=ax, color='skyblue', stat='density',
                 line_kws={'linewidth': 2, 'color': 'darkblue'})

    # Calculate mean and standard deviation for normal distribution
    mu, sigma = data.mean(), data.std()

    # Create x values for normal distribution curve
    x = np.linspace(data.min(), data.max(), 100)

    # Plot normal distribution curve
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r--', linewidth=2,
            label=f'Normal Dist: $\mu={mu:.2f}$, $\sigma={sigma:.2f}$')

    # Add mean and median lines
    ax.axvline(mu, color='red', linestyle='-', linewidth=1.5, label=f'Mean: {mu:.2f}')
    ax.axvline(data.median(), color='green', linestyle='--', linewidth=1.5,
               label=f'Median: {data.median():.2f}')

    # Add skewness and kurtosis annotation
    skewness = stats.skew(data.dropna())
    kurtosis = stats.kurtosis(data.dropna())
    stats_text = (f"Skewness: {skewness:.3f}\n"
                  f"Kurtosis: {kurtosis:.3f}\n"
                  f"n: {len(data.dropna())}")

    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add title and labels
    ax.set_title(f'Distribution of {variable_name}', fontsize=16)
    ax.set_xlabel(variable_name, fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.legend()

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)

    # Save the plot
    output_file = os.path.join(output_dir, f'{variable_name.replace(" ", "_").lower()}_distribution.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return output_file


def create_segment_distribution_plot(df: pd.DataFrame, variable: str, segment_var: str,
                                     output_dir: str = 'results/phase3/figures') -> str:
    """
    Create a distribution plot showing variable distribution across different segments.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    variable : str
        Name of the variable to plot
    segment_var : str
        Name of the segmentation variable
    output_dir : str
        Directory to save the plot

    Returns
    -------
    str
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get unique segments
    segments = df[segment_var].unique()

    # If too many segments, limit to the most common ones
    if len(segments) > 5:
        top_segments = df[segment_var].value_counts().nlargest(5).index
        df_plot = df[df[segment_var].isin(top_segments)]
        segments = top_segments
    else:
        df_plot = df

    # Plot KDE for each segment
    sns.kdeplot(data=df_plot, x=variable, hue=segment_var, ax=ax, common_norm=False, fill=True, alpha=0.4)

    # Add segment means as vertical lines
    for segment in segments:
        segment_mean = df[df[segment_var] == segment][variable].mean()
        ax.axvline(segment_mean, linestyle='--',
                   label=f'{segment_var}={segment} Mean: {segment_mean:.2f}')

    # Add title and labels
    ax.set_title(f'Distribution of {variable} by {segment_var}', fontsize=16)
    ax.set_xlabel(variable, fontsize=14)
    ax.set_ylabel('Density', fontsize=14)

    # Handle legend (might be cluttered with vertical lines)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 10:  # If too many items in legend
        # Keep only the KDE plot legends, not the vertical lines
        ax.legend(handles[:len(segments)], labels[:len(segments)], title=segment_var)
    else:
        ax.legend(title=segment_var)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)

    # Save the plot
    output_file = os.path.join(output_dir,
                               f'{variable.replace(" ", "_").lower()}_by_{segment_var.replace(" ", "_").lower()}.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return output_file


def analyze_variable_distribution(df: pd.DataFrame, variable: str) -> Dict:
    """
    Analyze the distribution of a single variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    variable : str
        Name of the variable to analyze

    Returns
    -------
    Dict
        Dictionary containing analysis results
    """
    results = {}

    # Basic statistics
    data = df[variable].dropna()
    results['count'] = len(data)
    results['missing'] = df[variable].isna().sum()
    results['mean'] = data.mean()
    results['median'] = data.median()
    results['std'] = data.std()
    results['min'] = data.min()
    results['max'] = data.max()
    results['range'] = data.max() - data.min()
    results['iqr'] = np.percentile(data, 75) - np.percentile(data, 25)
    results['skewness'] = stats.skew(data)
    results['kurtosis'] = stats.kurtosis(data)

    # Determine if distribution is approximately normal
    results['normality_tests'] = test_normality(data)
    is_normal = all([test.get('normal', False) for test in results['normality_tests'].values()
                     if 'normal' in test])
    results['is_normal'] = is_normal

    # Create distribution plot
    results['distribution_plot'] = create_distribution_plot(data, variable)

    # Create QQ plot
    results['qq_plot'] = create_qq_plot(data, variable)

    # Determine distribution type based on skewness and kurtosis
    skewness = results['skewness']
    kurtosis = results['kurtosis']

    if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
        results['distribution_type'] = "Approximately normal"
    elif skewness > 1.0:
        results['distribution_type'] = "Right-skewed (positive skew)"
    elif skewness < -1.0:
        results['distribution_type'] = "Left-skewed (negative skew)"
    elif kurtosis > 1.0:
        results['distribution_type'] = "Leptokurtic (heavy-tailed)"
    elif kurtosis < -1.0:
        results['distribution_type'] = "Platykurtic (light-tailed)"
    else:
        results['distribution_type'] = "Moderately non-normal"

    # Medical interpretation
    if variable == 'age':
        if abs(skewness) > 0.5:
            results[
                'medical_interpretation'] = "Age distribution shows skewness, indicating potential sampling bias or focus on specific age groups."
        else:
            results[
                'medical_interpretation'] = "Age distribution appears balanced, suggesting a representative sample across age groups."

    elif variable == 'chol':
        if skewness > 0.5:
            results[
                'medical_interpretation'] = "Cholesterol distribution is right-skewed, with some patients having unusually high values, which is common in cardiovascular studies."
        else:
            results[
                'medical_interpretation'] = "Cholesterol distribution is relatively symmetric, suggesting a well-distributed range of cholesterol levels."

    elif variable == 'thalach':
        if skewness < -0.5:
            results[
                'medical_interpretation'] = "Maximum heart rate distribution is left-skewed, suggesting more patients with higher maximum heart rates, possibly indicating a younger or more active sample."
        else:
            results[
                'medical_interpretation'] = "Maximum heart rate distribution shows expected pattern for a diverse cardiac patient population."

    elif variable == 'oldpeak':
        if skewness > 0.5:
            results[
                'medical_interpretation'] = "ST depression distribution is right-skewed, with most patients showing smaller depressions and fewer patients with severe ST depression."
        else:
            results[
                'medical_interpretation'] = "ST depression distribution is balanced, suggesting a diverse range of cardiac stress responses."

    else:
        results[
            'medical_interpretation'] = f"Distribution of {variable} provides insights into the spread and central tendency of this parameter in the patient population."

    return results


def analyze_segment_distributions(df: pd.DataFrame, variable: str, segment_var: str) -> Dict:
    """
    Analyze the distribution of a variable across different segments.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    variable : str
        Name of the variable to analyze
    segment_var : str
        Name of the segmentation variable

    Returns
    -------
    Dict
        Dictionary containing segment analysis results
    """
    results = {}

    # Get unique segments
    segments = df[segment_var].unique()
    results['segments'] = [str(s) for s in segments]

    # If segmentation variable is target_binary, use more descriptive labels
    if segment_var == 'target_binary':
        segment_labels = {0: 'No Disease', 1: 'Disease'}
    elif segment_var == 'sex':
        segment_labels = {0: 'Female', 1: 'Male'}
    elif segment_var == 'age_group':
        segment_labels = {str(s): str(s) for s in segments}  # Keep as is
    else:
        segment_labels = {str(s): str(s) for s in segments}

    # Create segment distribution plot
    results['segment_plot'] = create_segment_distribution_plot(df, variable, segment_var)

    # Calculate statistics for each segment
    segment_stats = {}
    for segment in segments:
        data = df[df[segment_var] == segment][variable].dropna()
        segment_label = segment_labels.get(segment, str(segment))

        segment_stats[segment_label] = {
            'count': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'skewness': stats.skew(data) if len(data) > 8 else None,
            'kurtosis': stats.kurtosis(data) if len(data) > 8 else None
        }

        # Only run normality tests if sufficient data
        if len(data) > 20:
            segment_stats[segment_label]['normality_tests'] = test_normality(data)

    results['segment_stats'] = segment_stats

    # Statistical test for difference between segments
    if len(segments) == 2 and segment_var in ['sex', 'target_binary']:
        # Two independent samples - use t-test or Mann-Whitney
        group1 = df[df[segment_var] == segments[0]][variable].dropna()
        group2 = df[df[segment_var] == segments[1]][variable].dropna()

        # Test for normal distribution
        _, p1 = stats.shapiro(group1) if len(group1) <= 5000 else (0, 0)
        _, p2 = stats.shapiro(group2) if len(group2) <= 5000 else (0, 0)

        if p1 > 0.05 and p2 > 0.05:
            # Both are approximately normal, use t-test
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            results['difference_test'] = {
                'test': 'Independent samples t-test',
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        else:
            # Not normal, use Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(group1, group2)
            results['difference_test'] = {
                'test': 'Mann-Whitney U test',
                'statistic': u_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

        # Effect size - Cohen's d
        mean1, std1 = group1.mean(), group1.std()
        mean2, std2 = group2.mean(), group2.std()

        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

        cohens_d = abs(mean1 - mean2) / pooled_std

        results['effect_size'] = {
            'method': "Cohen's d",
            'value': cohens_d,
            'interpretation': 'Small' if cohens_d < 0.5 else 'Medium' if cohens_d < 0.8 else 'Large'
        }

    elif len(segments) > 2:
        # Multiple groups - use ANOVA or Kruskal-Wallis
        groups = [df[df[segment_var] == segment][variable].dropna() for segment in segments]

        # Check if all groups have sufficient data
        if all(len(group) > 5 for group in groups):
            # Test for normality and equal variances
            normality = [stats.shapiro(group)[1] > 0.05 if len(group) <= 5000 else False for group in groups]

            if all(normality):
                # If all groups are approximately normal, use one-way ANOVA
                f_stat, p_value = stats.f_oneway(*groups)
                results['difference_test'] = {
                    'test': 'One-way ANOVA',
                    'statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

                # Calculate effect size (Eta-squared)
                # Between-group sum of squares
                grand_mean = df[variable].mean()
                segment_means = [group.mean() for group in groups]
                segment_counts = [len(group) for group in groups]

                ss_between = sum(count * (mean - grand_mean) ** 2 for count, mean in zip(segment_counts, segment_means))

                # Total sum of squares
                ss_total = sum((df[variable] - grand_mean) ** 2)

                eta_squared = ss_between / ss_total

                results['effect_size'] = {
                    'method': 'Eta-squared',
                    'value': eta_squared,
                    'interpretation': 'Small' if eta_squared < 0.06 else 'Medium' if eta_squared < 0.14 else 'Large'
                }

            else:
                # If not all groups are normal, use Kruskal-Wallis H-test
                h_stat, p_value = stats.kruskal(*groups)
                results['difference_test'] = {
                    'test': 'Kruskal-Wallis H-test',
                    'statistic': h_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

                # Effect size for Kruskal-Wallis (Eta-squared using H statistic)
                eta_h_squared = (h_stat - len(groups) + 1) / (len(df) - len(groups))
                results['effect_size'] = {
                    'method': 'Eta-squared (H)',
                    'value': eta_h_squared,
                    'interpretation': 'Small' if eta_h_squared < 0.06 else 'Medium' if eta_h_squared < 0.14 else 'Large'
                }

    # Medical interpretation
    if segment_var == 'target_binary' and 'difference_test' in results and results['difference_test']['significant']:
        if variable == 'age':
            results[
                'medical_interpretation'] = "Significant age differences between patients with and without heart disease suggest age is an important risk factor."
        elif variable == 'chol':
            results[
                'medical_interpretation'] = "Significant cholesterol differences between patients with and without heart disease confirm cholesterol's role in cardiovascular risk."
        elif variable == 'thalach':
            results[
                'medical_interpretation'] = "Significant differences in maximum heart rate between patients with and without heart disease may indicate functional cardiac capacity disparities."
        elif variable == 'oldpeak':
            results[
                'medical_interpretation'] = "Significant differences in ST depression between groups highlight its value as a diagnostic indicator."
        else:
            results[
                'medical_interpretation'] = f"Significant differences in {variable} between patients with and without heart disease suggest its potential diagnostic or prognostic value."

    elif segment_var == 'sex' and 'difference_test' in results and results['difference_test']['significant']:
        results[
            'medical_interpretation'] = f"Gender-based differences in {variable} may indicate the need for gender-specific risk assessment or treatment approaches."

    elif segment_var == 'age_group' and 'difference_test' in results and results['difference_test']['significant']:
        results[
            'medical_interpretation'] = f"Age-related variations in {variable} suggest the importance of age-stratified approaches to cardiac risk assessment."

    else:
        results[
            'medical_interpretation'] = f"Analysis of {variable} across different {segment_var} groups provides insights into population heterogeneity."

    return results


def perform_distribution_analysis(df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive distribution analysis on key variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset for analysis

    Returns
    -------
    Dict
        Dictionary containing all distribution analysis results
    """
    results = {}

    print("Performing probability distribution analysis...")

    # Define key variables to analyze
    key_variables = ['age', 'chol', 'thalach', 'oldpeak', 'trestbps']

    # Key segments to analyze
    key_segments = {
        'target_binary': 'Disease Status',
        'sex': 'Gender',
        'age_group': 'Age Group'
    }

    # Store individual variable analyses
    variable_distributions = {}
    for var in key_variables:
        if var in df.columns:
            print(f"\n1. Analyzing distribution of {var}...")
            variable_distributions[var] = analyze_variable_distribution(df, var)

    results['variable_distributions'] = variable_distributions

    # Store normality test results separately for easy access
    normality_tests = {}
    for var, analysis in variable_distributions.items():
        normality_tests[var] = analysis['normality_tests']

    results['normality_tests'] = normality_tests

    # Segment analysis
    segment_analyses = {}
    for var in key_variables:
        if var not in df.columns:
            continue

        var_segments = {}
        for segment_var, segment_name in key_segments.items():
            if segment_var in df.columns:
                print(f"\n2. Analyzing {var} distribution by {segment_name}...")
                var_segments[segment_var] = analyze_segment_distributions(df, var, segment_var)

        segment_analyses[var] = var_segments

    results['segment_analyses'] = segment_analyses

    # Collect key findings and clinical interpretations
    key_findings = []
    clinical_interpretations = []

    # Add findings from individual variable analyses
    for var, analysis in variable_distributions.items():
        distribution_type = analysis.get('distribution_type', '')
        key_findings.append(f"{var}: {distribution_type} distribution")

        if 'medical_interpretation' in analysis:
            clinical_interpretations.append(analysis['medical_interpretation'])

    # Add findings from segment analyses
    for var, segments in segment_analyses.items():
        for segment_var, analysis in segments.items():
            if 'difference_test' in analysis and analysis['difference_test']['significant']:
                segment_name = key_segments.get(segment_var, segment_var)
                test_name = analysis['difference_test']['test']
                p_value = analysis['difference_test']['p_value']
                key_findings.append(
                    f"{var} varies significantly across {segment_name} groups ({test_name}, p={p_value:.4f})")

                if 'medical_interpretation' in analysis:
                    clinical_interpretations.append(analysis['medical_interpretation'])

    results['key_findings'] = key_findings
    results['clinical_interpretations'] = clinical_interpretations

    print("\nProbability distribution analysis complete.")

    return results


if __name__ == "__main__":
    # Test the module if run directly
    print("Distribution Analysis Module for the Heart Disease dataset.")
    print("This module provides distribution analysis functions for the Heart Disease dataset.")
    print("Import and use in another script or notebook.")