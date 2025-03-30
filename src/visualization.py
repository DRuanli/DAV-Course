"""
Visualization Module for Heart Disease Dataset Analysis

This module provides standardized visualization functions for the
UCI Heart Disease dataset analysis.

Author: Team Member 3
Date: March 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import os


# Set up visualization style and parameters
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

    # Set color palette - a colorblind-friendly palette
    sns.set_palette("colorblind")

    # Create results/figures directory if it doesn't exist
    os.makedirs('results/figures', exist_ok=True)

    print("Visualization environment set up successfully.")


def plot_distribution(df: pd.DataFrame, variable: str, hue: Optional[str] = None,
                      bins: int = 30, save: bool = False) -> plt.Figure:
    """
    Create a distribution plot for a continuous variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    variable : str
        Name of the variable to plot
    hue : Optional[str]
        Name of categorical variable for grouping (default: None)
    bins : int
        Number of bins for histogram (default: 30)
    save : bool
        Whether to save the figure (default: False)

    Returns
    -------
    plt.Figure
        The created figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create KDE plot with histogram
    if hue is not None and hue in df.columns:
        # Plot with groups
        if hue == 'sex':
            # Map sex to meaningful labels
            df_plot = df.copy()
            df_plot['sex'] = df_plot['sex'].map({1: 'Male', 0: 'Female'})
            hue_var = 'sex'
        elif hue == 'num' and df[hue].nunique() > 2:
            # Create binary version of num for better visualization
            df_plot = df.copy()
            df_plot['heart_disease'] = df_plot['num'].apply(lambda x: 'Disease' if x > 0 else 'No Disease')
            hue_var = 'heart_disease'
        else:
            df_plot = df
            hue_var = hue

        sns.histplot(data=df_plot, x=variable, hue=hue_var, kde=True, bins=bins, alpha=0.6, ax=ax)
    else:
        # Simple plot without groups
        sns.histplot(data=df, x=variable, kde=True, bins=bins, alpha=0.6, ax=ax)

    # Add mean line
    mean_val = df[variable].mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')

    # Add median line
    median_val = df[variable].median()
    ax.axvline(median_val, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2f}')

    # Set title and labels
    ax.set_title(f'Distribution of {variable}')
    ax.set_xlabel(variable)
    ax.set_ylabel('Frequency')
    ax.legend()

    # Add descriptive statistics as text
    stats_text = (f"Mean: {df[variable].mean():.2f}\n"
                  f"Median: {df[variable].median():.2f}\n"
                  f"Std Dev: {df[variable].std():.2f}\n"
                  f"Min: {df[variable].min():.2f}\n"
                  f"Max: {df[variable].max():.2f}")

    plt.text(0.95, 0.95, stats_text, transform=ax.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save:
        filename = f"results/figures/{variable}_distribution.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")

    return fig


def plot_boxplot(df: pd.DataFrame, y_var: str, x_var: Optional[str] = None,
                 hue: Optional[str] = None, save: bool = False) -> plt.Figure:
    """
    Create a boxplot for a continuous variable, optionally grouped by a categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    y_var : str
        Name of the continuous variable for y-axis
    x_var : Optional[str]
        Name of the categorical variable for x-axis (default: None)
    hue : Optional[str]
        Name of categorical variable for additional grouping (default: None)
    save : bool
        Whether to save the figure (default: False)

    Returns
    -------
    plt.Figure
        The created figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create boxplot
    if x_var is not None:
        if x_var == 'num' and df[x_var].nunique() > 2:
            # Create labels for disease severity
            df_plot = df.copy()
            df_plot['disease_level'] = df_plot['num'].map({
                0: 'No Disease',
                1: 'Mild',
                2: 'Moderate',
                3: 'Severe',
                4: 'Very Severe'
            })
            sns.boxplot(data=df_plot, x='disease_level', y=y_var, hue=hue, ax=ax)
        else:
            sns.boxplot(data=df, x=x_var, y=y_var, hue=hue, ax=ax)
    else:
        sns.boxplot(data=df, y=y_var, ax=ax)

    # Add swarmplot for data points (limit to avoid overcrowding)
    if x_var is not None and len(df) <= 500:  # Only add points for smaller datasets
        if x_var == 'num' and df[x_var].nunique() > 2:
            sns.swarmplot(data=df_plot, x='disease_level', y=y_var, hue=hue,
                          dodge=True, color='black', alpha=0.5, ax=ax)
        else:
            sns.swarmplot(data=df, x=x_var, y=y_var, hue=hue,
                          dodge=True, color='black', alpha=0.5, ax=ax)

    # Set title and labels
    if x_var is not None:
        ax.set_title(f'Distribution of {y_var} by {x_var}')
    else:
        ax.set_title(f'Distribution of {y_var}')

    ax.set_xlabel(x_var if x_var is not None else '')
    ax.set_ylabel(y_var)

    plt.tight_layout()

    if save:
        if x_var is not None:
            filename = f"results/figures/{y_var}_by_{x_var}_boxplot.png"
        else:
            filename = f"results/figures/{y_var}_boxplot.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")

    return fig


def plot_correlation_heatmap(df: pd.DataFrame, method: str = 'pearson',
                             annot: bool = True, save: bool = False) -> plt.Figure:
    """
    Create a correlation heatmap for numeric variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing numeric variables
    method : str
        Correlation method ('pearson' or 'spearman') (default: 'pearson')
    annot : bool
        Whether to annotate cells with correlation values (default: True)
    save : bool
        Whether to save the figure (default: False)

    Returns
    -------
    plt.Figure
        The created figure object
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    # Calculate correlation matrix
    corr_matrix = numeric_df.corr(method=method)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create heatmap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=annot, fmt=".2f", square=True, linewidths=.5, ax=ax)

    # Set title
    ax.set_title(f'{method.capitalize()} Correlation Heatmap')

    plt.tight_layout()

    if save:
        filename = f"results/figures/{method}_correlation_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")

    return fig


def plot_categorical_counts(df: pd.DataFrame, variable: str, hue: Optional[str] = None,
                            save: bool = False) -> plt.Figure:
    """
    Create a count plot for a categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    variable : str
        Name of the categorical variable to plot
    hue : Optional[str]
        Name of categorical variable for grouping (default: None)
    save : bool
        Whether to save the figure (default: False)

    Returns
    -------
    plt.Figure
        The created figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create mapping for special variables
    df_plot = df.copy()

    # Special handling for common variables
    if variable == 'sex':
        df_plot['sex'] = df_plot['sex'].map({1: 'Male', 0: 'Female'})
    elif variable == 'cp':
        df_plot['cp'] = df_plot['cp'].map({
            1: 'Typical Angina',
            2: 'Atypical Angina',
            3: 'Non-anginal Pain',
            4: 'Asymptomatic'
        })
    elif variable == 'num':
        df_plot['num'] = df_plot['num'].map({
            0: 'No Disease',
            1: 'Mild',
            2: 'Moderate',
            3: 'Severe',
            4: 'Very Severe'
        })

    # Create count plot
    if hue is not None and hue in df.columns:
        if hue == 'sex':
            df_plot['sex'] = df_plot['sex'].map({1: 'Male', 0: 'Female'})
        sns.countplot(data=df_plot, x=variable, hue=hue, ax=ax)
    else:
        sns.countplot(data=df_plot, x=variable, ax=ax)

    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')

    # Set title and labels
    ax.set_title(f'Count of {variable}')
    ax.set_xlabel(variable)
    ax.set_ylabel('Count')

    # Rotate x labels if needed
    plt.xticks(rotation=45 if len(str(df_plot[variable].iloc[0])) > 5 else 0)

    plt.tight_layout()

    if save:
        filename = f"results/figures/{variable}_countplot.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")

    return fig


def plot_scatter(df: pd.DataFrame, x_var: str, y_var: str, hue: Optional[str] = None,
                 size: Optional[str] = None, fit_reg: bool = True, save: bool = False) -> plt.Figure:
    """
    Create a scatter plot between two continuous variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    x_var : str
        Name of the variable for x-axis
    y_var : str
        Name of the variable for y-axis
    hue : Optional[str]
        Name of categorical variable for color coding points (default: None)
    size : Optional[str]
        Name of variable for sizing points (default: None)
    fit_reg : bool
        Whether to fit and plot regression line (default: True)
    save : bool
        Whether to save the figure (default: False)

    Returns
    -------
    plt.Figure
        The created figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create plot df with mappings if needed
    df_plot = df.copy()
    hue_var = hue

    if hue == 'sex':
        df_plot['sex'] = df_plot['sex'].map({1: 'Male', 0: 'Female'})
        hue_var = 'sex'
    elif hue == 'num' and df[hue].nunique() > 2:
        df_plot['heart_disease'] = df_plot['num'].apply(lambda x: 'Disease' if x > 0 else 'No Disease')
        hue_var = 'heart_disease'

    # Create scatter plot
    if hue is not None and size is not None:
        sns.scatterplot(data=df_plot, x=x_var, y=y_var, hue=hue_var, size=size, alpha=0.7, ax=ax)
    elif hue is not None:
        sns.scatterplot(data=df_plot, x=x_var, y=y_var, hue=hue_var, alpha=0.7, ax=ax)
    else:
        sns.scatterplot(data=df_plot, x=x_var, y=y_var, alpha=0.7, ax=ax)

    # Add regression line if requested
    if fit_reg:
        if hue is not None:
            for category in df_plot[hue_var].unique():
                subset = df_plot[df_plot[hue_var] == category]
                sns.regplot(data=subset, x=x_var, y=y_var, scatter=False,
                            ax=ax, label=f'Trend for {category}')
        else:
            sns.regplot(data=df_plot, x=x_var, y=y_var, scatter=False,
                        ax=ax, label='Trend line')

    # Calculate and display correlation
    corr_coef = df[[x_var, y_var]].corr().iloc[0, 1]
    plt.text(0.95, 0.05, f'Correlation: {corr_coef:.2f}', transform=ax.transAxes,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Set title and labels
    ax.set_title(f'Relationship between {x_var} and {y_var}')
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)

    plt.tight_layout()

    if save:
        filename = f"results/figures/{x_var}_vs_{y_var}_scatter.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")

    return fig


def plot_pairplot(df: pd.DataFrame, vars: List[str], hue: Optional[str] = None,
                  diag_kind: str = 'kde', save: bool = False) -> sns.PairGrid:
    """
    Create a pairplot matrix for multiple variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    vars : List[str]
        List of variables to include in the pairplot
    hue : Optional[str]
        Name of categorical variable for grouping (default: None)
    diag_kind : str
        Kind of plot for diagonal elements ('kde' or 'hist') (default: 'kde')
    save : bool
        Whether to save the figure (default: False)

    Returns
    -------
    sns.PairGrid
        The created pairplot object
    """
    # Create plot df with mappings if needed
    df_plot = df.copy()
    hue_var = hue

    if hue == 'sex':
        df_plot['sex'] = df_plot['sex'].map({1: 'Male', 0: 'Female'})
        hue_var = 'sex'
    elif hue == 'num' and df[hue].nunique() > 2:
        df_plot['heart_disease'] = df_plot['num'].apply(lambda x: 'Disease' if x > 0 else 'No Disease')
        hue_var = 'heart_disease'

    # Create pairplot
    pairgrid = sns.pairplot(df_plot, vars=vars, hue=hue_var, diag_kind=diag_kind,
                            height=3, aspect=1.2)

    # Set title
    plt.suptitle('Pairwise Relationships', y=1.02, fontsize=20)
    plt.tight_layout()

    if save:
        filename = f"results/figures/pairplot_{'_'.join(vars)}.png"
        pairgrid.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")

    return pairgrid


def plot_violin(df: pd.DataFrame, y_var: str, x_var: Optional[str] = None,
                hue: Optional[str] = None, split: bool = False, save: bool = False) -> plt.Figure:
    """
    Create a violin plot for a continuous variable, optionally grouped by a categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    y_var : str
        Name of the continuous variable for y-axis
    x_var : Optional[str]
        Name of the categorical variable for x-axis (default: None)
    hue : Optional[str]
        Name of categorical variable for additional grouping (default: None)
    split : bool
        Whether to split the violins when hue is used (default: False)
    save : bool
        Whether to save the figure (default: False)

    Returns
    -------
    plt.Figure
        The created figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create plot df with mappings if needed
    df_plot = df.copy()

    # Special handling for common variables
    if x_var == 'sex':
        df_plot['sex'] = df_plot['sex'].map({1: 'Male', 0: 'Female'})
    elif x_var == 'num':
        df_plot['num'] = df_plot['num'].map({
            0: 'No Disease',
            1: 'Mild',
            2: 'Moderate',
            3: 'Severe',
            4: 'Very Severe'
        })

    if hue == 'sex':
        df_plot['sex'] = df_plot['sex'].map({1: 'Male', 0: 'Female'})

    # Create violin plot
    if x_var is not None:
        sns.violinplot(data=df_plot, x=x_var, y=y_var, hue=hue, split=split, ax=ax)
    else:
        sns.violinplot(data=df_plot, y=y_var, ax=ax)

    # Set title and labels
    if x_var is not None:
        ax.set_title(f'Distribution of {y_var} by {x_var}')
    else:
        ax.set_title(f'Distribution of {y_var}')

    ax.set_xlabel(x_var if x_var is not None else '')
    ax.set_ylabel(y_var)

    # Rotate x labels if needed
    if x_var is not None:
        plt.xticks(rotation=45 if len(df_plot[x_var].astype(str).iloc[0]) > 5 else 0)

    plt.tight_layout()

    if save:
        if x_var is not None:
            filename = f"results/figures/{y_var}_by_{x_var}_violin.png"
        else:
            filename = f"results/figures/{y_var}_violin.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")

    return fig


def create_standard_visualizations(df: pd.DataFrame, output_dir: str = 'results/figures') -> None:
    """
    Create a standard set of visualizations for the heart disease dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Heart disease dataset
    output_dir : str
        Directory to save figures (default: 'results/figures')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Setup visualization environment
    setup_visualization_env()

    # Create standard visualizations

    # 1. Age distribution
    plot_distribution(df, 'age', hue='num', save=True)

    # 2. Cholesterol distribution
    plot_distribution(df, 'chol', hue='num', save=True)

    # 3. Heart rate distribution
    plot_distribution(df, 'thalach', hue='num', save=True)

    # 4. ST depression distribution
    plot_distribution(df, 'oldpeak', hue='num', save=True)

    # 5. Blood pressure distribution
    plot_distribution(df, 'trestbps', hue='num', save=True)

    # 6. Correlation heatmap
    plot_correlation_heatmap(df, save=True)

    # 7. Chest pain type counts
    plot_categorical_counts(df, 'cp', hue='num', save=True)

    # 8. Gender counts
    plot_categorical_counts(df, 'sex', hue='num', save=True)

    # 9. Target distribution
    plot_categorical_counts(df, 'num', save=True)

    # 10. Age vs. Heart rate scatter
    plot_scatter(df, 'age', 'thalach', hue='num', save=True)

    # 11. Cholesterol by chest pain type boxplot
    plot_boxplot(df, 'chol', 'cp', save=True)

    # 12. Heart rate by gender violinplot
    plot_violin(df, 'thalach', 'sex', hue='num', save=True)

    # 13. Pairplot of key variables
    plot_pairplot(df, ['age', 'chol', 'thalach', 'oldpeak'], hue='num', save=True)

    print(f"Created and saved standard visualizations to {output_dir}")


if __name__ == "__main__":
    # Test the functions if run directly
    print("This module provides visualization functions for the Heart Disease dataset.")
    print("Import and use in another script or notebook.")