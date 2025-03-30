"""
Visualization Suite Module for Heart Disease Dataset Analysis

This module implements a comprehensive visualization suite for the UCI
Heart Disease dataset, creating various types of visualizations for EDA.

Author: Team Member 3
Date: March 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy import stats

# Define custom color palettes for consistency
DISEASE_PALETTE = ['#4878D0', '#EE854A']  # Blue for No Disease, Orange for Disease
CONTINUOUS_PALETTE = 'viridis'
CORRELATION_PALETTE = 'RdBu_r'
CATEGORICAL_PALETTE = 'Set2'


def setup_visualization_env():
    """
    Set up the visualization environment with standardized parameters.
    """
    # Set seaborn style
    sns.set(style="whitegrid")

    # Set matplotlib parameters
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 20

    # Create results/figures directory if it doesn't exist
    os.makedirs('results/figures', exist_ok=True)

    print("Visualization environment set up successfully.")


def create_histograms_for_continuous(df: pd.DataFrame, output_dir: str = 'results/figures') -> Dict[str, str]:
    """
    Create histograms for all continuous variables with normal curve overlay.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing continuous variables
    output_dir : str
        Directory to save figures (default: 'results/figures')

    Returns
    -------
    Dict[str, str]
        Dictionary mapping variable names to output file paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Identify continuous variables
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Filter out binary and categorical variables
    continuous_vars = [col for col in numeric_cols if df[col].nunique() > 5 and
                       col not in ['num', 'target_binary'] and
                       'missing' not in col and 'outlier' not in col]

    output_files = {}

    for var in continuous_vars:
        # Skip derived features if original exists
        if f"{var}_original" in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram with KDE
        sns.histplot(
            data=df,
            x=var,
            kde=True,
            bins=30,
            color='dodgerblue',
            line_kws={'linewidth': 2, 'color': 'darkblue'},
            ax=ax
        )

        # Add a normal curve for comparison
        mu, sigma = df[var].mean(), df[var].std()
        x = np.linspace(df[var].min(), df[var].max(), 100)
        y = stats.norm.pdf(x, mu, sigma) * len(df) * (df[var].max() - df[var].min()) / 30
        ax.plot(x, y, 'r--', linewidth=2, label=f'Normal: μ={mu:.2f}, σ={sigma:.2f}')

        # Add vertical lines for mean and median
        ax.axvline(df[var].mean(), color='red', linestyle='-', linewidth=1.5, label=f'Mean: {df[var].mean():.2f}')
        ax.axvline(df[var].median(), color='green', linestyle='--', linewidth=1.5,
                   label=f'Median: {df[var].median():.2f}')

        # Add statistical annotations
        stats_text = (
            f"Mean: {df[var].mean():.2f}\n"
            f"Median: {df[var].median():.2f}\n"
            f"Std Dev: {df[var].std():.2f}\n"
            f"Min: {df[var].min():.2f}\n"
            f"Max: {df[var].max():.2f}\n"
            f"Skewness: {df[var].skew():.2f}\n"
            f"Kurtosis: {df[var].kurtosis():.2f}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Enhance the plot
        ax.set_title(f'Distribution of {var}', fontsize=16)
        ax.set_xlabel(var, fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.legend(loc='upper left')
        plt.tight_layout()

        # Save the figure
        output_file = f"{output_dir}/{var}_histogram.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        output_files[var] = output_file

    print(f"Created {len(output_files)} histograms for continuous variables.")
    return output_files


def create_histograms_by_target(df: pd.DataFrame, output_dir: str = 'results/figures') -> Dict[str, str]:
    """
    Create histograms for continuous variables split by target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing continuous variables and target
    output_dir : str
        Directory to save figures (default: 'results/figures')

    Returns
    -------
    Dict[str, str]
        Dictionary mapping variable names to output file paths
    """
    # Ensure target variable exists
    if 'target_binary' not in df.columns and 'num' in df.columns:
        df['target_binary'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
    elif 'target_binary' not in df.columns:
        print("Target variable not found. Cannot create histograms by target.")
        return {}

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Identify continuous variables
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Filter out binary and categorical variables
    continuous_vars = [col for col in numeric_cols if df[col].nunique() > 5 and
                       col not in ['num', 'target_binary'] and
                       'missing' not in col and 'outlier' not in col]

    output_files = {}

    for var in continuous_vars:
        # Skip derived features if original exists
        if f"{var}_original" in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 7))

        # Split data by target
        no_disease = df[df['target_binary'] == 0][var].dropna()
        disease = df[df['target_binary'] == 1][var].dropna()

        # Plot histograms
        sns.histplot(no_disease, kde=True, color=DISEASE_PALETTE[0], label='No Disease', alpha=0.6, ax=ax)
        sns.histplot(disease, kde=True, color=DISEASE_PALETTE[1], label='Disease', alpha=0.6, ax=ax)

        # Add means as vertical lines
        ax.axvline(no_disease.mean(), color=DISEASE_PALETTE[0], linestyle='--',
                   linewidth=2, label=f'No Disease Mean: {no_disease.mean():.2f}')
        ax.axvline(disease.mean(), color=DISEASE_PALETTE[1], linestyle='--',
                   linewidth=2, label=f'Disease Mean: {disease.mean():.2f}')

        # Add statistical annotations
        t_stat, p_val = stats.ttest_ind(no_disease, disease, equal_var=False)

        stats_text = (
            f"No Disease: μ={no_disease.mean():.2f}, σ={no_disease.std():.2f}\n"
            f"Disease: μ={disease.mean():.2f}, σ={disease.std():.2f}\n"
            f"Difference: {disease.mean() - no_disease.mean():.2f}\n"
            f"T-test: t={t_stat:.2f}, p={p_val:.4f}\n"
            f"Significant: {'Yes' if p_val < 0.05 else 'No'}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Enhance the plot
        ax.set_title(f'Distribution of {var} by Disease Status', fontsize=16)
        ax.set_xlabel(var, fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.legend(loc='upper left')
        plt.tight_layout()

        # Save the figure
        output_file = f"{output_dir}/{var}_by_target_histogram.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        output_files[var] = output_file

    print(f"Created {len(output_files)} histograms split by target variable.")
    return output_files


def create_bar_charts_for_categorical(df: pd.DataFrame, output_dir: str = 'results/figures') -> Dict[str, str]:
    """
    Create bar charts for all categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing categorical variables
    output_dir : str
        Directory to save figures (default: 'results/figures')

    Returns
    -------
    Dict[str, str]
        Dictionary mapping variable names to output file paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Identify categorical variables
    categorical_vars = []

    # Traditional categorical columns
    categorical_vars.extend(df.select_dtypes(include=['object', 'category']).columns.tolist())

    # Numeric columns that are actually categorical
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    for col in numeric_cols:
        if df[col].nunique() <= 10 and col not in ['age', 'chol', 'trestbps', 'thalach',
                                                   'oldpeak'] and 'missing' not in col and 'outlier' not in col:
            categorical_vars.append(col)

    # Add derived categorical features
    categorical_vars.extend([col for col in df.columns if col.endswith('_category') or col.endswith('_group')])

    # Remove duplicates
    categorical_vars = list(set(categorical_vars))

    output_files = {}

    for var in categorical_vars:
        # Skip some variables
        if var in ['num', 'source'] or 'missing' in var or 'outlier' in var:
            continue

        fig, ax = plt.subplots(figsize=(12, 7))

        # Handle different types of categorical variables
        if var == 'sex':
            # Create a copy with mapped values
            temp_df = df.copy()
            temp_df['sex_label'] = temp_df['sex'].map({0: 'Female', 1: 'Male'})
            counts = temp_df['sex_label'].value_counts().sort_index()

            # Plot
            sns.barplot(x=counts.index, y=counts.values, palette=CATEGORICAL_PALETTE, ax=ax)

        elif var == 'cp':
            # Create a copy with mapped values
            temp_df = df.copy()
            temp_df['cp_label'] = temp_df['cp'].map({
                1: 'Typical Angina',
                2: 'Atypical Angina',
                3: 'Non-anginal Pain',
                4: 'Asymptomatic'
            })
            counts = temp_df['cp_label'].value_counts().sort_index()

            # Plot
            sns.barplot(x=counts.index, y=counts.values, palette=CATEGORICAL_PALETTE, ax=ax)
            plt.xticks(rotation=45, ha='right')

        elif var in ['slope', 'ca', 'thal']:
            # Create mappings for special variables
            if var == 'slope':
                labels = {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'}
            elif var == 'ca':
                labels = {0: '0 vessels', 1: '1 vessel', 2: '2 vessels', 3: '3 vessels'}
            elif var == 'thal':
                labels = {3: 'Normal', 6: 'Fixed Defect', 7: 'Reversible Defect'}

            # Create a copy with mapped values
            temp_df = df.copy()
            temp_df[f'{var}_label'] = temp_df[var].map(labels)
            counts = temp_df[f'{var}_label'].value_counts().sort_index()

            # Plot
            sns.barplot(x=counts.index, y=counts.values, palette=CATEGORICAL_PALETTE, ax=ax)
            if len(counts) > 4:
                plt.xticks(rotation=45, ha='right')

        else:
            # Regular categorical variable
            counts = df[var].value_counts().sort_index()

            # Plot
            sns.barplot(x=counts.index, y=counts.values, palette=CATEGORICAL_PALETTE, ax=ax)
            if len(counts) > 4:
                plt.xticks(rotation=45, ha='right')

        # Add count labels on top of bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom',
                        fontsize=12, fontweight='bold')

        # Enhance the plot
        ax.set_title(f'Distribution of {var}', fontsize=16)
        ax.set_xlabel(var, fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        plt.tight_layout()

        # Save the figure
        output_file = f"{output_dir}/{var}_barplot.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        output_files[var] = output_file

    print(f"Created {len(output_files)} bar charts for categorical variables.")
    return output_files


def create_bar_charts_by_target(df: pd.DataFrame, output_dir: str = 'results/figures') -> Dict[str, str]:
    """
    Create bar charts for categorical variables split by target.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing categorical variables and target
    output_dir : str
        Directory to save figures (default: 'results/figures')

    Returns
    -------
    Dict[str, str]
        Dictionary mapping variable names to output file paths
    """
    # Ensure target variable exists
    if 'target_binary' not in df.columns and 'num' in df.columns:
        df['target_binary'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
    elif 'target_binary' not in df.columns:
        print("Target variable not found. Cannot create bar charts by target.")
        return {}

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Identify categorical variables
    categorical_vars = []

    # Traditional categorical columns
    categorical_vars.extend(df.select_dtypes(include=['object', 'category']).columns.tolist())

    # Numeric columns that are actually categorical
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    for col in numeric_cols:
        if df[col].nunique() <= 10 and col not in ['age', 'chol', 'trestbps', 'thalach',
                                                   'oldpeak'] and 'missing' not in col and 'outlier' not in col:
            categorical_vars.append(col)

    # Add derived categorical features
    categorical_vars.extend([col for col in df.columns if col.endswith('_category') or col.endswith('_group')])

    # Remove duplicates and exclude target
    categorical_vars = list(set([var for var in categorical_vars if var != 'target_binary' and var != 'num']))

    output_files = {}

    for var in categorical_vars:
        # Skip some variables
        if var in ['source'] or 'missing' in var or 'outlier' in var:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Handle special variables
        if var == 'sex':
            # Create a mapping dictionary
            sex_map = {0: 'Female', 1: 'Male'}

            # Count plot
            sns.countplot(data=df, x='sex', hue='target_binary', palette=DISEASE_PALETTE, ax=ax1)
            ax1.set_xticklabels(['Female', 'Male'])

            # Percentage plot
            df_pct = pd.crosstab(df['sex'], df['target_binary'], normalize='index').reset_index()
            df_pct = pd.melt(df_pct, id_vars=['sex'], value_vars=[0, 1],
                             var_name='target_binary', value_name='percentage')
            df_pct['percentage'] = df_pct['percentage'] * 100
            df_pct['sex'] = df_pct['sex'].map(sex_map)

            sns.barplot(data=df_pct, x='sex', y='percentage', hue='target_binary', palette=DISEASE_PALETTE, ax=ax2)
            ax2.set_ylabel('Percentage (%)')

        elif var == 'cp':
            # Create a mapping dictionary
            cp_map = {1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-anginal Pain', 4: 'Asymptomatic'}

            # Add mapped column for better plotting
            df_temp = df.copy()
            df_temp['cp_label'] = df_temp['cp'].map(cp_map)

            # Count plot
            sns.countplot(data=df_temp, x='cp_label', hue='target_binary', palette=DISEASE_PALETTE, ax=ax1)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

            # Percentage plot
            df_pct = pd.crosstab(df['cp'], df['target_binary'], normalize='index').reset_index()
            df_pct = pd.melt(df_pct, id_vars=['cp'], value_vars=[0, 1],
                             var_name='target_binary', value_name='percentage')
            df_pct['percentage'] = df_pct['percentage'] * 100
            df_pct['cp'] = df_pct['cp'].map(cp_map)

            sns.barplot(data=df_pct, x='cp', y='percentage', hue='target_binary', palette=DISEASE_PALETTE, ax=ax2)
            ax2.set_ylabel('Percentage (%)')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

        else:
            # Count plot
            sns.countplot(data=df, x=var, hue='target_binary', palette=DISEASE_PALETTE, ax=ax1)
            if df[var].nunique() > 4:
                ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

            # Percentage plot
            df_pct = pd.crosstab(df[var], df['target_binary'], normalize='index').reset_index()
            df_pct = pd.melt(df_pct, id_vars=[var], value_vars=[0, 1],
                             var_name='target_binary', value_name='percentage')
            df_pct['percentage'] = df_pct['percentage'] * 100

            sns.barplot(data=df_pct, x=var, y='percentage', hue='target_binary', palette=DISEASE_PALETTE, ax=ax2)
            ax2.set_ylabel('Percentage (%)')
            if df[var].nunique() > 4:
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

        # Add chart titles
        ax1.set_title(f'Count of {var} by Disease Status', fontsize=14)
        ax2.set_title(f'Percentage of {var} by Disease Status', fontsize=14)

        # Update legend labels
        for ax in [ax1, ax2]:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, ['No Disease', 'Disease'])

        # Add chi-square test result
        contingency_table = pd.crosstab(df[var], df['target_binary'])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

        # Calculate Cramer's V
        n = contingency_table.sum().sum()
        cramer_v = np.sqrt(chi2 / (n * min(contingency_table.shape) - 1)) if n > 0 else 0

        stats_text = f"Chi-square: {chi2:.2f}, p-value: {p:.4f}\nCramer's V: {cramer_v:.4f}"
        fig.text(0.5, 0.01, stats_text, ha='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for the text at the bottom

        # Save the figure
        output_file = f"{output_dir}/{var}_by_target_barplot.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        output_files[var] = output_file

    print(f"Created {len(output_files)} bar charts split by target variable.")
    return output_files


def create_correlation_heatmap(df: pd.DataFrame, output_dir: str = 'results/figures') -> str:
    """
    Create correlation heatmap for numerical attributes.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing numerical attributes
    output_dir : str
        Directory to save figures (default: 'results/figures')

    Returns
    -------
    str
        Path to the saved heatmap file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Filter out derived columns
    numeric_cols = [col for col in numeric_cols if
                    'missing' not in col and 'outlier' not in col and '_original' not in col]

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create the heatmap figure
    plt.figure(figsize=(14, 12))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    heatmap = sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8}
    )

    plt.title('Correlation Matrix of Numerical Features', fontsize=18)
    plt.tight_layout()

    # Save the figure
    output_file = f"{output_dir}/correlation_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created correlation heatmap with {len(numeric_cols)} variables.")
    return output_file


def create_correlation_heatmap_with_target(df: pd.DataFrame, output_dir: str = 'results/figures') -> str:
    """
    Create correlation heatmap highlighting correlations with target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing numerical attributes and target
    output_dir : str
        Directory to save figures (default: 'results/figures')

    Returns
    -------
    str
        Path to the saved heatmap file
    """
    # Ensure target variable exists
    if 'target_binary' not in df.columns and 'num' in df.columns:
        df['target_binary'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
    elif 'target_binary' not in df.columns:
        print("Target variable not found. Using standard correlation heatmap.")
        return create_correlation_heatmap(df, output_dir)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Filter out derived columns
    numeric_cols = [col for col in numeric_cols if
                    'missing' not in col and 'outlier' not in col and '_original' not in col]

    # Make sure target_binary is in the list and at the end
    if 'target_binary' in numeric_cols:
        numeric_cols.remove('target_binary')
    numeric_cols.append('target_binary')

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()

    # Create the heatmap figure
    plt.figure(figsize=(14, 12))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap
    heatmap = sns.heatmap(
        corr_matrix,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8}
    )

    plt.title('Correlation Matrix with Target Variable', fontsize=18)
    plt.tight_layout()

    # Save the figure
    output_file = f"{output_dir}/correlation_with_target_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    # Create a focused correlation plot with target
    plt.figure(figsize=(10, 8))

    # Get correlations with target and sort
    target_corr = corr_matrix['target_binary'].drop('target_binary').sort_values(ascending=False)

    # Create horizontal bar chart
    colors = [DISEASE_PALETTE[1] if x > 0 else DISEASE_PALETTE[0] for x in target_corr]
    ax = target_corr.plot(kind='barh', color=colors, figsize=(10, 8))

    # Add correlation values
    for i, v in enumerate(target_corr):
        ax.text(v + 0.01 if v > 0 else v - 0.13, i, f'{v:.2f}',
                color='black', fontweight='bold', ha='left' if v > 0 else 'right', va='center')

    # Customize the plot
    plt.title('Correlation with Heart Disease (target_binary)', fontsize=16)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)

    # Save the figure
    target_output_file = f"{output_dir}/target_correlation_barplot.png"
    plt.savefig(target_output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created correlation heatmap and target correlation barplot.")
    return output_file


def create_boxplots(df: pd.DataFrame, output_dir: str = 'results/figures') -> Dict[str, str]:
    """
    Create boxplots for continuous variables comparing across disease status.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing continuous variables and target
    output_dir : str
        Directory to save figures (default: 'results/figures')

    Returns
    -------
    Dict[str, str]
        Dictionary mapping variable names to output file paths
    """
    # Ensure target variable exists
    if 'target_binary' not in df.columns and 'num' in df.columns:
        df['target_binary'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
    elif 'target_binary' not in df.columns:
        print("Target variable not found. Cannot create comparative boxplots.")
        return {}

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Identify continuous variables
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Filter out binary and categorical variables
    continuous_vars = [col for col in numeric_cols if df[col].nunique() > 10 and
                       col not in ['num', 'target_binary'] and
                       'missing' not in col and 'outlier' not in col]

    output_files = {}

    for var in continuous_vars:
        # Skip derived features if original exists
        if f"{var}_original" in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))

        # Create boxplot
        sns.boxplot(
            data=df,
            x='target_binary',
            y=var,
            palette=DISEASE_PALETTE,
            ax=ax
        )

        # Add swarmplot for data points (with smaller point size)
        sns.swarmplot(
            data=df,
            x='target_binary',
            y=var,
            color='black',
            alpha=0.5,
            size=3,
            ax=ax
        )

        # Add statistical annotation
        no_disease = df[df['target_binary'] == 0][var].dropna()
        disease = df[df['target_binary'] == 1][var].dropna()

        t_stat, p_val = stats.ttest_ind(no_disease, disease, equal_var=False)

        # Calculate effect size (Cohen's d)
        mean_diff = disease.mean() - no_disease.mean()
        pooled_std = np.sqrt(((len(disease) - 1) * disease.std() ** 2 +
                              (len(no_disease) - 1) * no_disease.std() ** 2) /
                             (len(disease) + len(no_disease) - 2))
        cohens_d = abs(mean_diff) / pooled_std

        effect_size_text = 'Small' if cohens_d < 0.5 else 'Medium' if cohens_d < 0.8 else 'Large'

        stats_text = (
            f"No Disease (n={len(no_disease)}): μ={no_disease.mean():.2f}\n"
            f"Disease (n={len(disease)}): μ={disease.mean():.2f}\n"
            f"Difference: {mean_diff:.2f} ({(mean_diff / no_disease.mean() * 100):.1f}%)\n"
            f"T-test: t={t_stat:.2f}, p={p_val:.4f}\n"
            f"Cohen's d: {cohens_d:.2f} ({effect_size_text} effect)"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Update x-axis labels
        ax.set_xticklabels(['No Disease', 'Disease'])

        # Enhance the plot
        ax.set_title(f'Distribution of {var} by Disease Status', fontsize=16)
        ax.set_xlabel('')
        ax.set_ylabel(var, fontsize=14)
        plt.tight_layout()

        # Save the figure
        output_file = f"{output_dir}/{var}_boxplot.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        output_files[var] = output_file

    print(f"Created {len(output_files)} boxplots comparing across disease status.")
    return output_files


def create_categorical_boxplots(df: pd.DataFrame, output_dir: str = 'results/figures') -> Dict[str, str]:
    """
    Create boxplots for continuous variables across categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing continuous and categorical variables
    output_dir : str
        Directory to save figures (default: 'results/figures')

    Returns
    -------
    Dict[str, str]
        Dictionary mapping variable combinations to output file paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Identify continuous variables
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    continuous_vars = [col for col in numeric_cols if df[col].nunique() > 10 and
                       'missing' not in col and 'outlier' not in col and '_original' not in col]

    # Identify important categorical variables
    categorical_vars = ['sex', 'cp', 'exang', 'fbs', 'restecg']
    if 'target_binary' in df.columns:
        categorical_vars.append('target_binary')

    # Add derived categorical columns
    categorical_vars.extend([col for col in df.columns if col.endswith('_category') or col.endswith('_group')])

    # Keep only existing columns
    categorical_vars = [col for col in categorical_vars if col in df.columns]

    output_files = {}

    # Select a subset of important continuous variables
    key_continuous = ['age', 'thalach', 'chol', 'trestbps', 'oldpeak']
    key_continuous = [var for var in key_continuous if var in continuous_vars]

    # Create boxplots for key combinations
    for cont_var in key_continuous:
        for cat_var in categorical_vars:
            # Skip if too many categories
            if df[cat_var].nunique() > 8:
                continue

            fig, ax = plt.subplots(figsize=(12, 7))

            # Handle special categorical variables
            if cat_var == 'sex':
                # Create a temporary dataframe with mapped sex
                temp_df = df.copy()
                temp_df['sex_label'] = temp_df['sex'].map({0: 'Female', 1: 'Male'})

                # Create boxplot
                sns.boxplot(data=temp_df, x='sex_label', y=cont_var, palette=CATEGORICAL_PALETTE, ax=ax)

                # Add statistical annotation
                female_data = df[df['sex'] == 0][cont_var].dropna()
                male_data = df[df['sex'] == 1][cont_var].dropna()

                t_stat, p_val = stats.ttest_ind(female_data, male_data, equal_var=False)

                stats_text = (
                    f"Female (n={len(female_data)}): μ={female_data.mean():.2f}\n"
                    f"Male (n={len(male_data)}): μ={male_data.mean():.2f}\n"
                    f"T-test: t={t_stat:.2f}, p={p_val:.4f}\n"
                    f"Significant: {'Yes' if p_val < 0.05 else 'No'}"
                )
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            elif cat_var == 'cp':
                # Create a temporary dataframe with mapped cp
                temp_df = df.copy()
                temp_df['cp_label'] = temp_df['cp'].map({
                    1: 'Typical Angina',
                    2: 'Atypical Angina',
                    3: 'Non-anginal Pain',
                    4: 'Asymptomatic'
                })

                # Create boxplot
                sns.boxplot(data=temp_df, x='cp_label', y=cont_var, palette=CATEGORICAL_PALETTE, ax=ax)
                plt.xticks(rotation=45, ha='right')

                # Add ANOVA result
                groups = [df[df['cp'] == cp_val][cont_var].dropna() for cp_val in sorted(df['cp'].unique())]
                f_stat, p_val = stats.f_oneway(*groups)

                stats_text = (
                    f"ANOVA: F={f_stat:.2f}, p={p_val:.4f}\n"
                    f"Significant: {'Yes' if p_val < 0.05 else 'No'}"
                )
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            elif cat_var == 'target_binary':
                # Create boxplot
                sns.boxplot(data=df, x=cat_var, y=cont_var, palette=DISEASE_PALETTE, ax=ax)

                # Update x-axis labels
                ax.set_xticklabels(['No Disease', 'Disease'])

                # Add statistical annotation
                no_disease = df[df['target_binary'] == 0][cont_var].dropna()
                disease = df[df['target_binary'] == 1][cont_var].dropna()

                t_stat, p_val = stats.ttest_ind(no_disease, disease, equal_var=False)

                stats_text = (
                    f"No Disease (n={len(no_disease)}): μ={no_disease.mean():.2f}\n"
                    f"Disease (n={len(disease)}): μ={disease.mean():.2f}\n"
                    f"T-test: t={t_stat:.2f}, p={p_val:.4f}\n"
                    f"Significant: {'Yes' if p_val < 0.05 else 'No'}"
                )
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            else:
                # Regular boxplot
                sns.boxplot(data=df, x=cat_var, y=cont_var, palette=CATEGORICAL_PALETTE, ax=ax)

                # Check if we need to rotate x-tick labels
                if df[cat_var].nunique() > 4:
                    plt.xticks(rotation=45, ha='right')

                # Add statistical annotation if appropriate
                if df[cat_var].nunique() > 1:
                    # ANOVA
                    groups = [df[df[cat_var] == val][cont_var].dropna() for val in sorted(df[cat_var].unique())]
                    valid_groups = [g for g in groups if len(g) > 0]

                    if len(valid_groups) > 1:
                        f_stat, p_val = stats.f_oneway(*valid_groups)

                        stats_text = (
                            f"ANOVA: F={f_stat:.2f}, p={p_val:.4f}\n"
                            f"Significant: {'Yes' if p_val < 0.05 else 'No'}"
                        )
                        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                                verticalalignment='top', horizontalalignment='left',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            # Enhance the plot
            ax.set_title(f'Distribution of {cont_var} by {cat_var}', fontsize=16)
            ax.set_xlabel(cat_var, fontsize=14)
            ax.set_ylabel(cont_var, fontsize=14)
            plt.tight_layout()

            # Save the figure
            output_file = f"{output_dir}/{cont_var}_by_{cat_var}_boxplot.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            output_files[f"{cont_var}_by_{cat_var}"] = output_file

    print(f"Created {len(output_files)} boxplots across categorical variables.")
    return output_files


def create_scatter_plots(df: pd.DataFrame, output_dir: str = 'results/figures') -> Dict[str, str]:
    """
    Create scatter plots for pairs of continuous variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing continuous variables
    output_dir : str
        Directory to save figures (default: 'results/figures')

    Returns
    -------
    Dict[str, str]
        Dictionary mapping variable pairs to output file paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Identify continuous variables
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    continuous_vars = [col for col in numeric_cols if df[col].nunique() > 10 and
                       'missing' not in col and 'outlier' not in col and '_original' not in col]

    # Select important pairs to visualize
    var_pairs = [
        ('age', 'thalach'),  # Age vs. Max Heart Rate
        ('age', 'chol'),  # Age vs. Cholesterol
        ('trestbps', 'chol'),  # Blood Pressure vs. Cholesterol
        ('thalach', 'oldpeak'),  # Max Heart Rate vs. ST Depression
        ('age', 'oldpeak')  # Age vs. ST Depression
    ]

    # Keep only pairs where both variables exist
    var_pairs = [(x, y) for x, y in var_pairs if x in continuous_vars and y in continuous_vars]

    output_files = {}

    for x_var, y_var in var_pairs:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Check if target variable exists for color coding
        if 'target_binary' in df.columns:
            # Scatter plot with disease status color coding
            scatter = sns.scatterplot(
                data=df,
                x=x_var,
                y=y_var,
                hue='target_binary',
                palette=DISEASE_PALETTE,
                s=70,
                alpha=0.7,
                ax=ax
            )

            # Update legend
            handles, labels = scatter.get_legend_handles_labels()
            ax.legend(handles, ['No Disease', 'Disease'], title='Disease Status')

            # Add regression lines for each group
            sns.regplot(data=df[df['target_binary'] == 0], x=x_var, y=y_var,
                        scatter=False, color=DISEASE_PALETTE[0], line_kws={'linestyle': '--'}, ax=ax)
            sns.regplot(data=df[df['target_binary'] == 1], x=x_var, y=y_var,
                        scatter=False, color=DISEASE_PALETTE[1], line_kws={'linestyle': '--'}, ax=ax)

            # Calculate correlations by group
            corr_no_disease = df[df['target_binary'] == 0][[x_var, y_var]].corr().iloc[0, 1]
            corr_disease = df[df['target_binary'] == 1][[x_var, y_var]].corr().iloc[0, 1]

            stats_text = (
                f"No Disease: r = {corr_no_disease:.2f}\n"
                f"Disease: r = {corr_disease:.2f}\n"
                f"Overall: r = {df[[x_var, y_var]].corr().iloc[0, 1]:.2f}"
            )

        else:
            # Simple scatter plot without grouping
            scatter = sns.scatterplot(
                data=df,
                x=x_var,
                y=y_var,
                s=70,
                alpha=0.7,
                ax=ax
            )

            # Add regression line
            sns.regplot(data=df, x=x_var, y=y_var,
                        scatter=False, line_kws={'linestyle': '--'}, ax=ax)

            # Calculate correlation
            corr = df[[x_var, y_var]].corr().iloc[0, 1]

            stats_text = f"Correlation: r = {corr:.2f}"

        # Add correlation text
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Enhance the plot
        ax.set_title(f'Relationship between {x_var} and {y_var}', fontsize=16)
        ax.set_xlabel(x_var, fontsize=14)
        ax.set_ylabel(y_var, fontsize=14)
        plt.tight_layout()

        # Save the figure
        output_file = f"{output_dir}/{x_var}_vs_{y_var}_scatter.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        output_files[f"{x_var}_vs_{y_var}"] = output_file

    print(f"Created {len(output_files)} scatter plots.")
    return output_files


def create_pairplot(df: pd.DataFrame, output_dir: str = 'results/figures') -> str:
    """
    Create a pairplot for key continuous variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing continuous variables
    output_dir : str
        Directory to save figures (default: 'results/figures')

    Returns
    -------
    str
        Path to the saved pairplot file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Select key variables for pairplot
    key_vars = ['age', 'thalach', 'chol', 'trestbps', 'oldpeak']
    # Keep only columns that exist in the dataframe
    key_vars = [var for var in key_vars if var in df.columns]

    # Include target if it exists
    if 'target_binary' in df.columns:
        pairplot_vars = key_vars + ['target_binary']
        hue = 'target_binary'
    else:
        pairplot_vars = key_vars
        hue = None

    # Create pairplot
    g = sns.pairplot(
        df[pairplot_vars],
        hue=hue,
        palette=DISEASE_PALETTE if hue else None,
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 50},
        diag_kws={'alpha': 0.6},
        height=2.5,
        aspect=1
    )

    # Update legend if applicable
    if hue:
        g._legend.set_title('Disease Status')
        for t, l in zip(g._legend.texts, ['No Disease', 'Disease']):
            t.set_text(l)

    # Enhance the plot
    g.fig.suptitle('Pairwise Relationships Between Key Variables', y=1.02, fontsize=18)
    plt.tight_layout()

    # Save the figure
    output_file = f"{output_dir}/key_variables_pairplot.png"
    g.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(g.fig)

    print(f"Created pairplot with {len(key_vars)} key variables.")
    return output_file


def create_visualization_dashboard(df: pd.DataFrame, output_dir: str = 'results/figures') -> str:
    """
    Create a comprehensive visualization dashboard.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset for visualization
    output_dir : str
        Directory to save figures (default: 'results/figures')

    Returns
    -------
    str
        Path to the saved dashboard file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a large figure for the dashboard
    fig = plt.figure(figsize=(20, 16))

    # Create a grid for the plots
    grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

    # 1. Age distribution by target
    ax1 = fig.add_subplot(grid[0, 0])
    if 'target_binary' in df.columns:
        for i, (target, color) in enumerate(zip([0, 1], DISEASE_PALETTE)):
            data = df[df['target_binary'] == target]['age']
            sns.kdeplot(data, fill=True, color=color, alpha=0.5,
                        label='No Disease' if target == 0 else 'Disease', ax=ax1)
    else:
        sns.kdeplot(df['age'], fill=True, color='blue', alpha=0.5, ax=ax1)
    ax1.set_title('Age Distribution by Disease Status')
    ax1.set_xlabel('Age')
    ax1.legend()

    # 2. Cholesterol boxplot by target
    ax2 = fig.add_subplot(grid[0, 1])
    if 'target_binary' in df.columns:
        sns.boxplot(data=df, x='target_binary', y='chol', palette=DISEASE_PALETTE, ax=ax2)
        ax2.set_xticklabels(['No Disease', 'Disease'])
    else:
        sns.boxplot(data=df, y='chol', color='blue', ax=ax2)
    ax2.set_title('Cholesterol by Disease Status')
    ax2.set_xlabel('')

    # 3. Chest pain type by target
    ax3 = fig.add_subplot(grid[0, 2])
    if 'cp' in df.columns:
        # Create a temporary dataframe with mapped cp
        temp_df = df.copy()
        temp_df['cp_label'] = temp_df['cp'].map({
            1: 'Typical Angina',
            2: 'Atypical Angina',
            3: 'Non-anginal Pain',
            4: 'Asymptomatic'
        })

        if 'target_binary' in df.columns:
            # Calculate percentages
            cp_pct = pd.crosstab(df['cp'], df['target_binary'], normalize='index').reset_index()
            cp_pct = pd.melt(cp_pct, id_vars=['cp'], value_vars=[0, 1],
                             var_name='target_binary', value_name='percentage')
            cp_pct['percentage'] = cp_pct['percentage'] * 100
            cp_pct['cp'] = cp_pct['cp'].map({
                1: 'Typical Angina',
                2: 'Atypical Angina',
                3: 'Non-anginal Pain',
                4: 'Asymptomatic'
            })

            sns.barplot(data=cp_pct, x='cp', y='percentage', hue='target_binary',
                        palette=DISEASE_PALETTE, ax=ax3)
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
            ax3.set_ylabel('Percentage (%)')
            handles, labels = ax3.get_legend_handles_labels()
            ax3.legend(handles, ['No Disease', 'Disease'])
        else:
            sns.countplot(data=temp_df, x='cp_label', ax=ax3)
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.set_title('Chest Pain Type Distribution')
        ax3.set_xlabel('')

    # 4. Age vs. Max Heart Rate scatter
    ax4 = fig.add_subplot(grid[1, 0])
    if 'thalach' in df.columns:
        if 'target_binary' in df.columns:
            sns.scatterplot(data=df, x='age', y='thalach', hue='target_binary',
                            palette=DISEASE_PALETTE, alpha=0.7, ax=ax4)
            handles, labels = ax4.get_legend_handles_labels()
            ax4.legend(handles, ['No Disease', 'Disease'])
        else:
            sns.scatterplot(data=df, x='age', y='thalach', alpha=0.7, ax=ax4)

        # Add regression line
        sns.regplot(data=df, x='age', y='thalach', scatter=False,
                    line_kws={'linestyle': '--', 'color': 'black'}, ax=ax4)

        ax4.set_title('Age vs. Maximum Heart Rate')
        ax4.set_xlabel('Age')
        ax4.set_ylabel('Maximum Heart Rate')

    # 5. Correlation heatmap
    ax5 = fig.add_subplot(grid[1, 1:])
    # Select important variables for correlation
    corr_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    if 'target_binary' in df.columns:
        corr_vars.append('target_binary')

    # Keep only columns that exist in the dataframe
    corr_vars = [var for var in corr_vars if var in df.columns]

    # Calculate correlation matrix
    corr_matrix = df[corr_vars].corr()

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        cmap='RdBu_r',
        vmax=1,
        vmin=-1,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=.5,
        cbar_kws={"shrink": .8},
        ax=ax5
    )
    ax5.set_title('Correlation Matrix of Key Variables')

    # 6. Max Heart Rate by Disease
    ax6 = fig.add_subplot(grid[2, 0])
    if 'thalach' in df.columns and 'target_binary' in df.columns:
        sns.violinplot(data=df, x='target_binary', y='thalach', palette=DISEASE_PALETTE, ax=ax6)
        ax6.set_xticklabels(['No Disease', 'Disease'])
        ax6.set_title('Maximum Heart Rate by Disease Status')
        ax6.set_xlabel('')
        ax6.set_ylabel('Maximum Heart Rate')

    # 7. ST Depression by Disease
    ax7 = fig.add_subplot(grid[2, 1])
    if 'oldpeak' in df.columns and 'target_binary' in df.columns:
        sns.violinplot(data=df, x='target_binary', y='oldpeak', palette=DISEASE_PALETTE, ax=ax7)
        ax7.set_xticklabels(['No Disease', 'Disease'])
        ax7.set_title('ST Depression by Disease Status')
        ax7.set_xlabel('')
        ax7.set_ylabel('ST Depression')

    # 8. Gender distribution by Disease
    ax8 = fig.add_subplot(grid[2, 2])
    if 'sex' in df.columns:
        # Create a temporary dataframe with mapped sex
        temp_df = df.copy()
        temp_df['sex_label'] = temp_df['sex'].map({0: 'Female', 1: 'Male'})

        if 'target_binary' in df.columns:
            # Calculate percentages
            sex_pct = pd.crosstab(df['sex'], df['target_binary'], normalize='index').reset_index()
            sex_pct = pd.melt(sex_pct, id_vars=['sex'], value_vars=[0, 1],
                              var_name='target_binary', value_name='percentage')
            sex_pct['percentage'] = sex_pct['percentage'] * 100
            sex_pct['sex'] = sex_pct['sex'].map({0: 'Female', 1: 'Male'})

            sns.barplot(data=sex_pct, x='sex', y='percentage', hue='target_binary',
                        palette=DISEASE_PALETTE, ax=ax8)
            ax8.set_ylabel('Percentage (%)')
            handles, labels = ax8.get_legend_handles_labels()
            ax8.legend(handles, ['No Disease', 'Disease'])
        else:
            sns.countplot(data=temp_df, x='sex_label', ax=ax8)
        ax8.set_title('Gender Distribution')
        ax8.set_xlabel('')

    # Add title to the figure
    fig.suptitle('Heart Disease Dataset Visualization Dashboard', fontsize=24, y=0.98)

    # Save the dashboard
    output_file = f"{output_dir}/visualization_dashboard.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Created comprehensive visualization dashboard.")
    return output_file


def create_all_visualizations(df: pd.DataFrame, output_dir: str = 'results/figures') -> Dict[str, Dict[str, str]]:
    """
    Create all visualization types for the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset for visualization
    output_dir : str
        Directory to save figures (default: 'results/figures')

    Returns
    -------
    Dict[str, Dict[str, str]]
        Dictionary mapping visualization types to their output files
    """
    # Ensure visualization environment is set up
    setup_visualization_env()

    # Create all visualization types
    visualizations = {}

    print("\nCreating histograms for continuous variables...")
    visualizations['histograms'] = create_histograms_for_continuous(df, output_dir)

    print("\nCreating histograms split by target variable...")
    visualizations['histograms_by_target'] = create_histograms_by_target(df, output_dir)

    print("\nCreating bar charts for categorical variables...")
    visualizations['bar_charts'] = create_bar_charts_for_categorical(df, output_dir)

    print("\nCreating bar charts split by target variable...")
    visualizations['bar_charts_by_target'] = create_bar_charts_by_target(df, output_dir)

    print("\nCreating correlation heatmap...")
    visualizations['correlation_heatmap'] = create_correlation_heatmap(df, output_dir)

    print("\nCreating correlation heatmap with target variable...")
    visualizations['correlation_with_target'] = create_correlation_heatmap_with_target(df, output_dir)

    print("\nCreating boxplots comparing across disease status...")
    visualizations['boxplots'] = create_boxplots(df, output_dir)

    print("\nCreating boxplots across categorical variables...")
    visualizations['categorical_boxplots'] = create_categorical_boxplots(df, output_dir)

    print("\nCreating scatter plots...")
    visualizations['scatter_plots'] = create_scatter_plots(df, output_dir)

    print("\nCreating pairplot...")
    visualizations['pairplot'] = create_pairplot(df, output_dir)

    print("\nCreating visualization dashboard...")
    visualizations['dashboard'] = create_visualization_dashboard(df, output_dir)

    print(f"\nAll visualizations created and saved to {output_dir}")

    return visualizations


if __name__ == "__main__":
    # Test the module if run directly
    print("Heart Disease Dataset Visualization Suite")
    print("This module provides visualization functions for the Heart Disease dataset.")
    print("Import and use in another script or notebook.")