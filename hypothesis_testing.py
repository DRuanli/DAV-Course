"""
Hypothesis Testing Module for Heart Disease Dataset

This module formulates and tests meaningful hypotheses on the UCI Heart
Disease dataset, including t-tests, Chi-square tests, and ANOVA.

Author: Team Member 2
Date: March 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Union
import logging
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Set up the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def t_test_continuous_variables(df: pd.DataFrame, variable: str, group_var: str = 'target_binary') -> Dict:
    """
    Perform t-test to compare means of a continuous variable between two groups.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    variable : str
        Name of the continuous variable to test
    group_var : str
        Name of the grouping variable (default: 'target_binary')

    Returns
    -------
    Dict
        Dictionary containing test results
    """
    # Check if variables exist in the dataset
    if variable not in df.columns:
        return {'error': f"Variable '{variable}' not found in dataset"}
    if group_var not in df.columns:
        return {'error': f"Grouping variable '{group_var}' not found in dataset"}

    # Check if the grouping variable has exactly 2 groups
    groups = df[group_var].unique()
    if len(groups) != 2:
        return {'error': f"Grouping variable must have exactly 2 groups, found {len(groups)}"}

    # Prepare the result dictionary
    result = {}

    # Set up null and alternative hypotheses
    result['null_hypothesis'] = f"There is no difference in the mean {variable} between the two groups of {group_var}"
    result[
        'alternative_hypothesis'] = f"There is a difference in the mean {variable} between the two groups of {group_var}"

    # Get the data for each group
    group1 = df[df[group_var] == groups[0]][variable].dropna()
    group2 = df[df[group_var] == groups[1]][variable].dropna()

    # Calculate basic statistics
    result['sample_sizes'] = {str(groups[0]): len(group1), str(groups[1]): len(group2)}
    result['means'] = {str(groups[0]): group1.mean(), str(groups[1]): group2.mean()}
    result['std_devs'] = {str(groups[0]): group1.std(), str(groups[1]): group2.std()}

    # Test for normality
    _, p1 = stats.shapiro(group1) if len(group1) <= 5000 else (0, 0)
    _, p2 = stats.shapiro(group2) if len(group2) <= 5000 else (0, 0)
    result['normality_tests'] = {
        str(groups[0]): {'shapiro_p_value': p1, 'normal': p1 > 0.05},
        str(groups[1]): {'shapiro_p_value': p2, 'normal': p2 > 0.05}
    }

    # Determine which test to use based on normality
    if p1 > 0.05 and p2 > 0.05:
        # Both groups are approximately normal, use t-test
        # Use Welch's t-test (unequal variances) for robustness
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

        result['test_used'] = "Welch's t-test (assuming unequal variances)"
        result['statistic'] = t_stat
        result['p_value'] = p_value
        result['significant'] = p_value < 0.05

        # Calculate confidence interval
        # Using Welch-Satterthwaite equation for degrees of freedom
        s1_squared = group1.var()
        s2_squared = group2.var()
        n1, n2 = len(group1), len(group2)

        # Degrees of freedom
        df_val = ((s1_squared / n1 + s2_squared / n2) ** 2) / \
                 ((s1_squared / n1) ** 2 / (n1 - 1) + (s2_squared / n2) ** 2 / (n2 - 1))

        # Standard error
        se = np.sqrt(s1_squared / n1 + s2_squared / n2)

        # t-critical value
        t_crit = stats.t.ppf(0.975, df_val)  # 95% CI

        # Mean difference
        mean_diff = group1.mean() - group2.mean()

        # Confidence interval
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se

        result['mean_difference'] = mean_diff
        result['confidence_interval'] = {
            'lower': ci_lower,
            'upper': ci_upper,
            'level': 0.95
        }

    else:
        # At least one group is not normal, use Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(group1, group2)

        result['test_used'] = "Mann-Whitney U test (non-parametric)"
        result['statistic'] = u_stat
        result['p_value'] = p_value
        result['significant'] = p_value < 0.05

        # For Mann-Whitney, we can use median and IQR instead of mean and CI
        result['medians'] = {str(groups[0]): group1.median(), str(groups[1]): group2.median()}
        result['iqrs'] = {
            str(groups[0]): np.percentile(group1, 75) - np.percentile(group1, 25),
            str(groups[1]): np.percentile(group2, 75) - np.percentile(group2, 25)
        }
        result['median_difference'] = group1.median() - group2.median()

    # Calculate effect size (Cohen's d)
    mean1, std1 = group1.mean(), group1.std()
    mean2, std2 = group2.mean(), group2.std()

    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

    cohens_d = abs(mean1 - mean2) / pooled_std

    result['effect_size'] = {
        'method': "Cohen's d",
        'value': cohens_d,
        'interpretation': 'Small' if cohens_d < 0.5 else 'Medium' if cohens_d < 0.8 else 'Large'
    }

    # Interpret the result
    if result['significant']:
        if group_var == 'target_binary':
            group_labels = {0: 'non-disease', 1: 'disease'}
        elif group_var == 'sex':
            group_labels = {0: 'female', 1: 'male'}
        else:
            group_labels = {groups[0]: str(groups[0]), groups[1]: str(groups[1])}

        group1_label = group_labels.get(groups[0], str(groups[0]))
        group2_label = group_labels.get(groups[1], str(groups[1]))

        if mean1 > mean2:
            result[
                'conclusion'] = f"The mean {variable} is significantly higher in the {group1_label} group compared to the {group2_label} group (p={p_value:.4f})"
        else:
            result[
                'conclusion'] = f"The mean {variable} is significantly lower in the {group1_label} group compared to the {group2_label} group (p={p_value:.4f})"
    else:
        result[
            'conclusion'] = f"There is no significant difference in {variable} between the two groups (p={p_value:.4f})"

    # Create visualization
    create_t_test_visualization(group1, group2, variable, groups, result)

    return result


def create_t_test_visualization(group1: pd.Series, group2: pd.Series, variable: str,
                                groups: np.ndarray, result: Dict, output_dir: str = 'results/phase3/figures'):
    """
    Create visualization for t-test results.

    Parameters
    ----------
    group1 : pd.Series
        Data for group 1
    group2 : pd.Series
        Data for group 2
    variable : str
        Name of the variable
    groups : np.ndarray
        Group labels
    result : Dict
        Dictionary containing test results
    output_dir : str
        Directory to save the visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Boxplot with swarmplot overlay
    if len(groups) == 2:
        # Create a new DataFrame for plotting
        plot_df = pd.DataFrame({
            'Group': [str(groups[0])] * len(group1) + [str(groups[1])] * len(group2),
            variable: pd.concat([group1, group2])
        })

        # For target_binary or sex, use more descriptive labels
        if groups[0] == 0 and groups[1] == 1:
            if len(plot_df['Group'].unique()) == 2:
                if len(group1) > 0 and len(group1) / (len(group1) + len(group2)) > 0.25:
                    plot_df['Group'] = plot_df['Group'].map({'0': 'No Disease', '1': 'Disease'})

        # Create boxplot
        sns.boxplot(x='Group', y=variable, data=plot_df, ax=ax1, palette='pastel')

        # Add individual points (limited to avoid overcrowding)
        if len(plot_df) < 500:
            sns.swarmplot(x='Group', y=variable, data=plot_df, ax=ax1, color='black', alpha=0.5, size=3)
        else:
            # Subsample for larger datasets
            subsample = plot_df.sample(n=min(500, len(plot_df)))
            sns.stripplot(x='Group', y=variable, data=subsample, ax=ax1, color='black', alpha=0.3, size=3, jitter=True)

    # Violin plot with means marked
    if len(groups) == 2:
        sns.violinplot(x='Group', y=variable, data=plot_df, ax=ax2, inner='quartile', palette='pastel')

        # Add mean lines
        mean1, mean2 = result['means'][str(groups[0])], result['means'][str(groups[1])]
        ax2.axhline(mean1, ls='--', color='red', alpha=0.6, xmin=0.1, xmax=0.3,
                    label=f'Mean (Group 1): {mean1:.2f}')
        ax2.axhline(mean2, ls='--', color='blue', alpha=0.6, xmin=0.7, xmax=0.9,
                    label=f'Mean (Group 2): {mean2:.2f}')
        ax2.legend()

    # Add title and test details
    fig.suptitle(f"Comparison of {variable} Between Groups", fontsize=16)

    test_used = result.get('test_used', 'Statistical test')
    p_value = result.get('p_value', 0)
    effect_size = result.get('effect_size', {}).get('value', 0)
    effect_interp = result.get('effect_size', {}).get('interpretation', '')

    test_details = (f"{test_used}\n"
                    f"p-value: {p_value:.4f} ({'Significant' if p_value < 0.05 else 'Not Significant'})\n"
                    f"Effect Size: {effect_size:.3f} ({effect_interp})")

    fig.text(0.5, 0.01, test_details, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make space for the text

    # Save figure
    output_file = os.path.join(output_dir, f'{variable}_group_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Add file path to result
    result['visualization'] = output_file


def chi_square_test(df: pd.DataFrame, var1: str, var2: str = 'target_binary') -> Dict:
    """
    Perform Chi-square test to examine association between two categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    var1 : str
        Name of the first categorical variable
    var2 : str
        Name of the second categorical variable (default: 'target_binary')

    Returns
    -------
    Dict
        Dictionary containing test results
    """
    # Check if variables exist in the dataset
    if var1 not in df.columns:
        return {'error': f"Variable '{var1}' not found in dataset"}
    if var2 not in df.columns:
        return {'error': f"Variable '{var2}' not found in dataset"}

    # Prepare the result dictionary
    result = {}

    # Set up null and alternative hypotheses
    result['null_hypothesis'] = f"There is no association between {var1} and {var2}"
    result['alternative_hypothesis'] = f"There is an association between {var1} and {var2}"

    # Create the contingency table
    contingency_table = pd.crosstab(df[var1], df[var2])
    result['contingency_table'] = contingency_table.to_dict()

    # Check if the contingency table has sufficient entries
    expected = stats.chi2_contingency(contingency_table)[3]
    if (expected < 5).any():
        result[
            'warning'] = "Some expected frequencies are less than 5, which may affect the validity of the chi-square test"

    # Perform the chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    result['test_used'] = "Pearson's Chi-square test"
    result['statistic'] = chi2
    result['p_value'] = p_value
    result['degrees_of_freedom'] = dof
    result['significant'] = p_value < 0.05

    # Calculate effect size (Cramer's V)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1  # Smaller dimension - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    result['effect_size'] = {
        'method': "Cramer's V",
        'value': cramers_v,
        'interpretation': 'Small' if cramers_v < 0.3 else 'Medium' if cramers_v < 0.5 else 'Large'
    }

    # Calculate percentages for the contingency table
    percentage_table = pd.crosstab(df[var1], df[var2], normalize='index') * 100
    result['percentage_table'] = percentage_table.to_dict()

    # Interpret the result
    if result['significant']:
        if var2 == 'target_binary':
            result[
                'conclusion'] = f"There is a significant association between {var1} and heart disease status (p={p_value:.4f})"
            if cramers_v >= 0.3:
                result['conclusion'] += f" with a {result['effect_size']['interpretation'].lower()} effect size"
        else:
            result['conclusion'] = f"There is a significant association between {var1} and {var2} (p={p_value:.4f})"
            if cramers_v >= 0.3:
                result['conclusion'] += f" with a {result['effect_size']['interpretation'].lower()} effect size"
    else:
        result['conclusion'] = f"There is no significant association between {var1} and {var2} (p={p_value:.4f})"

    # Create visualization
    create_chi_square_visualization(df, var1, var2, result)

    return result


def create_chi_square_visualization(df: pd.DataFrame, var1: str, var2: str,
                                    result: Dict, output_dir: str = 'results/phase3/figures'):
    """
    Create visualization for chi-square test results.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    var1 : str
        Name of the first categorical variable
    var2 : str
        Name of the second categorical variable
    result : Dict
        Dictionary containing test results
    output_dir : str
        Directory to save the visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Handle special cases for variable names
    var1_label = var1
    var2_label = var2

    if var2 == 'target_binary':
        var2_label = 'Disease Status'
        # Create new mapped column for plotting
        df = df.copy()
        df['Disease Status'] = df['target_binary'].map({0: 'No Disease', 1: 'Disease'})
        var2 = 'Disease Status'

    if var1 == 'sex':
        var1_label = 'Gender'
        # Create new mapped column for plotting
        df = df.copy()
        df['Gender'] = df['sex'].map({0: 'Female', 1: 'Male'})
        var1 = 'Gender'

    # Count plot
    sns.countplot(x=var1, hue=var2, data=df, ax=ax1, palette='pastel')
    ax1.set_title(f"Count of {var1_label} by {var2_label}", fontsize=14)
    ax1.set_xlabel(var1_label, fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)

    # Get legend labels for use in the percentage plot
    handles, labels = ax1.get_legend_handles_labels()

    # Percentage plot
    # Calculate the percentage within each category of var1
    percentage_data = pd.crosstab(df[var1], df[var2], normalize='index') * 100
    percentage_data = percentage_data.reset_index().melt(id_vars=var1, var_name=var2, value_name='Percentage')

    sns.barplot(x=var1, y='Percentage', hue=var2, data=percentage_data, ax=ax2, palette='pastel')
    ax2.set_title(f"Percentage of {var2_label} within each {var1_label} Category", fontsize=14)
    ax2.set_xlabel(var1_label, fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)

    # Use the same legend labels as the count plot
    ax2.legend(handles, labels)

    # Add chi-square test details
    chi2 = result.get('statistic', 0)
    p_value = result.get('p_value', 0)
    dof = result.get('degrees_of_freedom', 0)
    cramers_v = result.get('effect_size', {}).get('value', 0)
    effect_interp = result.get('effect_size', {}).get('interpretation', '')

    test_details = (f"Chi-square Test Results:\n"
                    f"χ² = {chi2:.2f}, df = {dof}, p-value = {p_value:.4f}\n"
                    f"Cramer's V = {cramers_v:.3f} ({effect_interp} effect size)")

    fig.text(0.5, 0.01, test_details, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make space for the text

    # Save figure
    var1_name = var1_label.replace(' ', '_').lower()
    var2_name = var2_label.replace(' ', '_').lower()
    output_file = os.path.join(output_dir, f'{var1_name}_{var2_name}_association.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Add file path to result
    result['visualization'] = output_file


def anova_test(df: pd.DataFrame, continuous_var: str, category_var: str) -> Dict:
    """
    Perform One-way ANOVA to test for differences in means across multiple groups.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    continuous_var : str
        Name of the continuous variable
    category_var : str
        Name of the categorical variable defining the groups

    Returns
    -------
    Dict
        Dictionary containing test results
    """
    # Check if variables exist in the dataset
    if continuous_var not in df.columns:
        return {'error': f"Variable '{continuous_var}' not found in dataset"}
    if category_var not in df.columns:
        return {'error': f"Variable '{category_var}' not found in dataset"}

    # Get unique categories
    categories = df[category_var].unique()

    # Need at least 3 categories for ANOVA
    if len(categories) < 3:
        return {'error': f"ANOVA requires at least 3 groups, found {len(categories)} in {category_var}"}

    # Prepare the result dictionary
    result = {}

    # Set up null and alternative hypotheses
    result[
        'null_hypothesis'] = f"There is no difference in the means of {continuous_var} across the different groups of {category_var}"
    result[
        'alternative_hypothesis'] = f"At least one group mean of {continuous_var} is different from the others across {category_var}"

    # Prepare data for each group
    groups = []
    group_data = {}
    for category in categories:
        data = df[df[category_var] == category][continuous_var].dropna()
        if len(data) > 0:
            groups.append(data)
            group_data[str(category)] = {
                'count': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max()
            }

    result['group_statistics'] = group_data

    # Test normality assumptions for each group
    normality_results = {}
    for category, data in group_data.items():
        group_data = df[df[category_var] == float(category) if category.replace('.', '', 1).isdigit()
        else category][continuous_var].dropna()
        if len(group_data) > 8:  # Need at least 8 data points for Shapiro-Wilk
            _, p_value = stats.shapiro(group_data)
            normality_results[category] = {'p_value': p_value, 'normal': p_value > 0.05}

    result['normality_tests'] = normality_results

    # Test homogeneity of variances
    variances = [group.var() for group in groups]
    _, levene_p = stats.levene(*groups)
    result['homogeneity_of_variances'] = {
        'test': "Levene's test",
        'p_value': levene_p,
        'equal_variances': levene_p > 0.05
    }

    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    result['test_used'] = "One-way ANOVA"
    result['statistic'] = f_stat
    result['p_value'] = p_value
    result['significant'] = p_value < 0.05

    # Degrees of freedom
    df_between = len(groups) - 1
    df_within = sum(len(group) - 1 for group in groups)
    df_total = df_between + df_within

    result['degrees_of_freedom'] = {
        'between_groups': df_between,
        'within_groups': df_within,
        'total': df_total
    }

    # Calculate effect size (Eta-squared)
    # Need to calculate the sum of squares
    grand_mean = np.concatenate(groups).mean()

    # Between-group sum of squares
    ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)

    # Total sum of squares
    ss_total = sum((val - grand_mean) ** 2 for group in groups for val in group)

    eta_squared = ss_between / ss_total

    result['effect_size'] = {
        'method': 'Eta-squared',
        'value': eta_squared,
        'interpretation': 'Small' if eta_squared < 0.06 else 'Medium' if eta_squared < 0.14 else 'Large'
    }

    # Perform post-hoc tests if ANOVA is significant
    if p_value < 0.05:
        # Create a DataFrame for post-hoc tests
        posthoc_df = pd.DataFrame({
            continuous_var: np.concatenate(groups),
            category_var: np.concatenate([[i] * len(group) for i, group in enumerate(groups)])
        })

        # Tukey's HSD test
        tukey = pairwise_tukeyhsd(posthoc_df[continuous_var], posthoc_df[category_var])

        # Convert Tukey results to dictionary
        tukey_results = []

        # Fixed: Handle the Tukey HSD results properly
        # Get the mapping of group indices to original category values
        group_mapping = {i: cat for i, cat in enumerate(categories)}

        # Process each comparison from the Tukey test
        for i in range(len(tukey.pvalues)):
            # Extract the group indices from MultiComparison object properly
            group1_idx = int(tukey.data.iloc[tukey.reject.index[i]]['group1'])
            group2_idx = int(tukey.data.iloc[tukey.reject.index[i]]['group2'])

            # Map indices back to original category values
            group1 = group_mapping[group1_idx]
            group2 = group_mapping[group2_idx]

            # Extract the other values for this comparison
            meandiff = tukey.meandiffs[i]
            p_adj = tukey.pvalues[i]
            lower = tukey.confint[i, 0]
            upper = tukey.confint[i, 1]
            reject = tukey.reject[i]

            tukey_results.append({
                'group1': str(group1),
                'group2': str(group2),
                'mean_difference': meandiff,
                'p_value': p_adj,
                'ci_lower': lower,
                'ci_upper': upper,
                'significant': reject
            })

        result['post_hoc_tests'] = {
            'method': "Tukey's HSD",
            'results': tukey_results
        }

        # Find significant pairwise comparisons
        significant_pairs = [f"{res['group1']} vs {res['group2']}" for res in tukey_results if res['significant']]
        result['significant_pairs'] = significant_pairs

    # Interpret the result
    if result['significant']:
        result[
            'conclusion'] = f"There are significant differences in {continuous_var} across the different groups of {category_var} (p={p_value:.4f})"
        if 'significant_pairs' in result and result['significant_pairs']:
            result[
                'conclusion'] += f". Significant differences were found between the following pairs: {', '.join(result['significant_pairs'])}"
    else:
        result[
            'conclusion'] = f"There are no significant differences in {continuous_var} across the different groups of {category_var} (p={p_value:.4f})"

    # Create visualization
    create_anova_visualization(df, continuous_var, category_var, result)

    return result


def create_anova_visualization(df: pd.DataFrame, continuous_var: str, category_var: str,
                               result: Dict, output_dir: str = 'results/phase3/figures'):
    """
    Create visualization for ANOVA test results.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    continuous_var : str
        Name of the continuous variable
    category_var : str
        Name of the categorical variable
    result : Dict
        Dictionary containing test results
    output_dir : str
        Directory to save the visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Create mapped variables for better display if needed
    df_plot = df.copy()

    # Boxplot
    sns.boxplot(x=category_var, y=continuous_var, data=df_plot, ax=ax1, palette='pastel')
    ax1.set_title(f"Distribution of {continuous_var} by {category_var}", fontsize=14)
    ax1.set_xlabel(category_var, fontsize=12)
    ax1.set_ylabel(continuous_var, fontsize=12)

    # If there are many categories, rotate x labels
    if len(df[category_var].unique()) > 4:
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # Means plot with error bars
    means = df_plot.groupby(category_var)[continuous_var].mean()
    errors = df_plot.groupby(category_var)[continuous_var].std()

    x_pos = np.arange(len(means))
    ax2.bar(x_pos, means, yerr=errors, align='center', alpha=0.7, capsize=10, color='lightblue', ecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(means.index)
    ax2.set_title(f"Mean {continuous_var} by {category_var} with Error Bars", fontsize=14)
    ax2.set_xlabel(category_var, fontsize=12)
    ax2.set_ylabel(f"Mean {continuous_var}", fontsize=12)

    # If there are many categories, rotate x labels
    if len(means) > 4:
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    # Add ANOVA test details
    f_stat = result.get('statistic', 0)
    p_value = result.get('p_value', 0)
    df_between = result.get('degrees_of_freedom', {}).get('between_groups', 0)
    df_within = result.get('degrees_of_freedom', {}).get('within_groups', 0)
    eta_squared = result.get('effect_size', {}).get('value', 0)
    effect_interp = result.get('effect_size', {}).get('interpretation', '')

    test_details = (f"One-way ANOVA Results:\n"
                    f"F({df_between}, {df_within}) = {f_stat:.2f}, p-value = {p_value:.4f}\n"
                    f"Eta-squared = {eta_squared:.3f} ({effect_interp} effect size)")

    # Add post-hoc test results if available
    if 'post_hoc_tests' in result and 'significant_pairs' in result and result['significant_pairs']:
        posthoc_details = f"\nSignificant differences between: {', '.join(result['significant_pairs'][:3])}"
        if len(result['significant_pairs']) > 3:
            posthoc_details += f" and {len(result['significant_pairs']) - 3} more pairs"
        test_details += posthoc_details

    fig.text(0.5, 0.01, test_details, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make space for the text

    # Save figure
    cont_name = continuous_var.replace(' ', '_').lower()
    cat_name = category_var.replace(' ', '_').lower()
    output_file = os.path.join(output_dir, f'{cont_name}_by_{cat_name}_anova.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Add file path to result
    result['visualization'] = output_file


def perform_hypothesis_testing(df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive hypothesis testing on the heart disease dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset for analysis

    Returns
    -------
    Dict
        Dictionary containing all hypothesis testing results
    """
    results = {}

    print("Performing hypothesis testing...")

    # 1. Test differences in continuous variables between disease/non-disease groups
    continuous_tests = {}

    key_continuous_vars = ['age', 'chol', 'thalach', 'oldpeak', 'trestbps']
    for var in key_continuous_vars:
        if var in df.columns and 'target_binary' in df.columns:
            print(f"\n1. Testing hypothesis: {var} differs between disease and non-disease groups")
            test_name = f"{var}_disease_difference"
            continuous_tests[test_name] = t_test_continuous_variables(df, var, 'target_binary')

    results['continuous_variable_tests'] = continuous_tests

    # 2. Test associations between categorical variables and disease
    categorical_tests = {}

    key_categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    for var in key_categorical_vars:
        if var in df.columns and 'target_binary' in df.columns:
            print(f"\n2. Testing hypothesis: {var} is associated with heart disease")
            test_name = f"{var}_disease_association"
            categorical_tests[test_name] = chi_square_test(df, var, 'target_binary')

    results['categorical_association_tests'] = categorical_tests

    # 3. Test differences across multiple groups (ANOVA)
    anova_tests = {}

    # Define pairs of variables for ANOVA tests
    anova_pairs = [
        ('age', 'cp'),  # Age differences across chest pain types
        ('thalach', 'cp'),  # Max heart rate differences across chest pain types
        ('chol', 'age_group'),  # Cholesterol differences across age groups
        ('thalach', 'age_group'),  # Max heart rate differences across age groups
        ('oldpeak', 'cp')  # ST depression differences across chest pain types
    ]

    for cont_var, cat_var in anova_pairs:
        if cont_var in df.columns and cat_var in df.columns:
            if len(df[cat_var].unique()) >= 3:  # Need at least 3 groups for ANOVA
                print(f"\n3. Testing hypothesis: {cont_var} differs across {cat_var} groups")
                test_name = f"{cont_var}_by_{cat_var}"
                anova_tests[test_name] = anova_test(df, cont_var, cat_var)

    results['anova_tests'] = anova_tests

    # Combine all test results into a single dictionary for easier access
    all_tests = {}
    all_tests.update({k: v for k, v in continuous_tests.items() if 'error' not in v})
    all_tests.update({k: v for k, v in categorical_tests.items() if 'error' not in v})
    all_tests.update({k: v for k, v in anova_tests.items() if 'error' not in v})

    results['test_results'] = all_tests

    # Collect clinical interpretations
    clinical_interpretations = []

    # Get significant findings from t-tests
    for test_name, test_result in continuous_tests.items():
        if test_result.get('significant', False):
            if 'conclusion' in test_result:
                if 'age' in test_name:
                    clinical_interpretations.append(
                        f"Age is a significant factor in heart disease, with {test_result['conclusion'].lower()}")
                elif 'chol' in test_name:
                    clinical_interpretations.append(
                        f"Cholesterol level is significantly related to heart disease presence, with {test_result['conclusion'].lower()}")
                elif 'thalach' in test_name:
                    clinical_interpretations.append(
                        f"Maximum heart rate achieved shows significant differences with heart disease status, suggesting functional cardiac capacity is an important indicator.")
                elif 'oldpeak' in test_name:
                    clinical_interpretations.append(
                        f"ST depression induced by exercise (oldpeak) is a strong indicator of heart disease, with {test_result['conclusion'].lower()}")
                else:
                    clinical_interpretations.append(test_result['conclusion'])

    # Get significant findings from chi-square tests
    for test_name, test_result in categorical_tests.items():
        if test_result.get('significant', False):
            if 'conclusion' in test_result:
                if 'sex' in test_name:
                    clinical_interpretations.append(
                        f"Gender is significantly associated with heart disease, suggesting different risk profiles between men and women.")
                elif 'cp' in test_name:
                    clinical_interpretations.append(
                        f"Chest pain type shows strong association with heart disease, highlighting its importance as a diagnostic indicator.")
                elif 'ca' in test_name or 'thal' in test_name:
                    clinical_interpretations.append(
                        f"The strong association between {test_name.split('_')[0]} and heart disease confirms its value in cardiac assessment.")
                else:
                    clinical_interpretations.append(test_result['conclusion'])

    # Get significant findings from ANOVA tests
    for test_name, test_result in anova_tests.items():
        if test_result.get('significant', False):
            if 'conclusion' in test_result:
                if 'chol_by_age_group' in test_name:
                    clinical_interpretations.append(
                        f"Cholesterol levels vary significantly across age groups, suggesting age-specific reference ranges may be appropriate.")
                elif 'thalach_by_age_group' in test_name:
                    clinical_interpretations.append(
                        f"Maximum heart rate decreases significantly with age, consistent with expected physiological changes.")
                elif 'oldpeak_by_cp' in test_name:
                    clinical_interpretations.append(
                        f"ST depression varies significantly by chest pain type, with asymptomatic patients often showing more severe ECG changes.")
                else:
                    clinical_interpretations.append(test_result['conclusion'])

    results['clinical_interpretations'] = clinical_interpretations

    print("\nHypothesis testing complete.")

    return results


if __name__ == "__main__":
    # Test the module if run directly
    print("Hypothesis Testing Module for the Heart Disease dataset.")
    print("This module provides hypothesis testing functions for the Heart Disease dataset.")
    print("Import and use in another script or notebook.")