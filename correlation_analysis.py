"""
Correlation Analysis Module for Heart Disease Dataset

This module implements comprehensive correlation analysis for the UCI
Heart Disease dataset, including Pearson, Spearman, and partial correlations.

Author: Team Member 3
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
import pingouin as pg
from sklearn.preprocessing import StandardScaler
from itertools import combinations

# Set up the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_pearson_correlations(df: pd.DataFrame,
                                   variables: List[str] = None,
                                   calculate_p_values: bool = True) -> Dict:
    """
    Calculate Pearson correlation coefficients between numerical variables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    variables : List[str], optional
        List of variables to include in the correlation analysis.
        If None, use all numeric columns
    calculate_p_values : bool, optional
        Whether to calculate p-values for correlations

    Returns
    -------
    Dict
        Dictionary containing correlation matrix and p-values if requested
    """
    # If variables not specified, use all numeric columns
    if variables is None:
        # Get numeric columns, excluding those that are likely categorical
        # or derivative of other columns
        variables = df.select_dtypes(include=['number']).columns.tolist()

        # Exclude some columns
        variables = [col for col in variables if not col.endswith('_missing')
                     and not col.endswith('_outlier') and not col.endswith('_original')
                     and col != 'source']

    # Only include variables that exist in the dataframe
    variables = [var for var in variables if var in df.columns]

    # Calculate correlation matrix
    corr_matrix = df[variables].corr(method='pearson')

    result = {
        'correlation_matrix': corr_matrix.to_dict()
    }

    # Calculate p-values if requested
    if calculate_p_values:
        p_values = pd.DataFrame(np.ones_like(corr_matrix),
                                index=corr_matrix.index,
                                columns=corr_matrix.columns)

        # Calculate p-values for each pair of variables
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i < j:  # Only calculate for unique pairs
                    # Get the data
                    data1 = df[var1].dropna()
                    data2 = df[var2].dropna()

                    # Get the common indices
                    common_idx = data1.index.intersection(data2.index)

                    # Calculate correlation and p-value if enough data
                    if len(common_idx) > 2:
                        r, p = stats.pearsonr(data1[common_idx], data2[common_idx])
                        p_values.loc[var1, var2] = p
                        p_values.loc[var2, var1] = p

        result['p_values'] = p_values.to_dict()

    return result


def calculate_spearman_correlations(df: pd.DataFrame,
                                    variables: List[str] = None,
                                    calculate_p_values: bool = True) -> Dict:
    """
    Calculate Spearman rank correlation coefficients between numerical variables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    variables : List[str], optional
        List of variables to include in the correlation analysis.
        If None, use all numeric columns
    calculate_p_values : bool, optional
        Whether to calculate p-values for correlations

    Returns
    -------
    Dict
        Dictionary containing correlation matrix and p-values if requested
    """
    # If variables not specified, use all numeric columns
    if variables is None:
        # Get numeric columns, excluding those that are likely categorical
        # or derivative of other columns
        variables = df.select_dtypes(include=['number']).columns.tolist()

        # Exclude some columns
        variables = [col for col in variables if not col.endswith('_missing')
                     and not col.endswith('_outlier') and not col.endswith('_original')
                     and col != 'source']

    # Only include variables that exist in the dataframe
    variables = [var for var in variables if var in df.columns]

    # Calculate correlation matrix
    corr_matrix = df[variables].corr(method='spearman')

    result = {
        'correlation_matrix': corr_matrix.to_dict()
    }

    # Calculate p-values if requested
    if calculate_p_values:
        p_values = pd.DataFrame(np.ones_like(corr_matrix),
                                index=corr_matrix.index,
                                columns=corr_matrix.columns)

        # Calculate p-values for each pair of variables
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i < j:  # Only calculate for unique pairs
                    # Get the data
                    data1 = df[var1].dropna()
                    data2 = df[var2].dropna()

                    # Get the common indices
                    common_idx = data1.index.intersection(data2.index)

                    # Calculate correlation and p-value if enough data
                    if len(common_idx) > 2:
                        r, p = stats.spearmanr(data1[common_idx], data2[common_idx])
                        p_values.loc[var1, var2] = p
                        p_values.loc[var2, var1] = p

        result['p_values'] = p_values.to_dict()

    return result


def calculate_partial_correlations(df: pd.DataFrame,
                                   variables: List[str] = None,
                                   control_variables: List[str] = ['age', 'sex']) -> Dict:
    """
    Calculate partial correlations controlling for specified variables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    variables : List[str], optional
        List of variables to include in the correlation analysis.
        If None, use all numeric columns
    control_variables : List[str], optional
        List of variables to control for (default: ['age', 'sex'])

    Returns
    -------
    Dict
        Dictionary containing partial correlation results
    """
    # If variables not specified, use all numeric columns except control variables
    if variables is None:
        # Get numeric columns, excluding those that are likely categorical
        # or derivative of other columns
        variables = df.select_dtypes(include=['number']).columns.tolist()

        # Exclude control variables and some columns
        variables = [col for col in variables if col not in control_variables
                     and not col.endswith('_missing') and not col.endswith('_outlier')
                     and not col.endswith('_original') and col != 'source']

    # Only include variables that exist in the dataframe
    variables = [var for var in variables if var in df.columns]
    control_variables = [var for var in control_variables if var in df.columns]

    # Check if we have any control variables
    if not control_variables:
        return {'error': "No valid control variables found"}

    # Check if pingouin is available (for partial correlations)
    try:
        # Initialize result dictionaries
        partial_corr_matrix = pd.DataFrame(np.zeros((len(variables), len(variables))),
                                           index=variables, columns=variables)
        partial_p_values = pd.DataFrame(np.ones((len(variables), len(variables))),
                                        index=variables, columns=variables)

        # Calculate partial correlations for each pair of variables
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i < j:  # Only calculate for unique pairs
                    try:
                        # Calculate partial correlation
                        partial_corr = pg.partial_corr(
                            data=df, x=var1, y=var2, covar=control_variables
                        )

                        # Extract r and p-value
                        r = partial_corr['r'].values[0]
                        p = partial_corr['p-val'].values[0]

                        # Store in matrices
                        partial_corr_matrix.loc[var1, var2] = r
                        partial_corr_matrix.loc[var2, var1] = r
                        partial_p_values.loc[var1, var2] = p
                        partial_p_values.loc[var2, var1] = p
                    except Exception as e:
                        # Skip this pair if calculation fails
                        logging.warning(f"Error calculating partial correlation for {var1} and {var2}: {e}")

        # Diagonal elements should be 1
        for var in variables:
            partial_corr_matrix.loc[var, var] = 1.0

        result = {
            'correlation_matrix': partial_corr_matrix.to_dict(),
            'p_values': partial_p_values.to_dict(),
            'control_variables': control_variables
        }

        return result

    except ImportError:
        # If pingouin is not available, use manual calculation
        logging.warning("Pingouin package not available, using manual partial correlation calculation")

        # Create a copy of the data with only the relevant variables
        all_vars = variables + control_variables
        data = df[all_vars].dropna()

        # Standardize the data
        scaler = StandardScaler()
        data_std = pd.DataFrame(scaler.fit_transform(data), columns=all_vars, index=data.index)

        # Initialize result dictionaries
        partial_corr_matrix = pd.DataFrame(np.zeros((len(variables), len(variables))),
                                           index=variables, columns=variables)

        # Calculate partial correlations using matrix algebra approach
        # (simplified version, not ideal for many control variables)
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i < j:  # Only calculate for unique pairs
                    try:
                        # Residuals after regressing out control variables for var1
                        X1 = data_std[control_variables]
                        y1 = data_std[var1]
                        beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
                        residual1 = y1 - X1.dot(beta1)

                        # Residuals after regressing out control variables for var2
                        X2 = data_std[control_variables]
                        y2 = data_std[var2]
                        beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
                        residual2 = y2 - X2.dot(beta2)

                        # Correlation between residuals is the partial correlation
                        r, p = stats.pearsonr(residual1, residual2)

                        # Store in matrices
                        partial_corr_matrix.loc[var1, var2] = r
                        partial_corr_matrix.loc[var2, var1] = r
                    except Exception as e:
                        # Skip this pair if calculation fails
                        logging.warning(f"Error calculating partial correlation for {var1} and {var2}: {e}")

        # Diagonal elements should be 1
        for var in variables:
            partial_corr_matrix.loc[var, var] = 1.0

        result = {
            'correlation_matrix': partial_corr_matrix.to_dict(),
            'control_variables': control_variables,
            'warning': "Pingouin package not available, using simplified calculation"
        }

        return result


def create_correlation_heatmap(correlation_matrix: pd.DataFrame,
                               p_values: pd.DataFrame = None,
                               title: str = "Correlation Heatmap",
                               output_file: str = "correlation_heatmap.png",
                               output_dir: str = 'results/phase3/figures',
                               cmap: str = 'coolwarm',
                               annotate: bool = True) -> str:
    """
    Create and save a correlation heatmap with significance indicators.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        Correlation matrix
    p_values : pd.DataFrame, optional
        P-values for the correlations
    title : str, optional
        Title for the heatmap
    output_file : str, optional
        Filename for the saved heatmap
    output_dir : str, optional
        Directory to save the heatmap
    cmap : str, optional
        Colormap for the heatmap
    annotate : bool, optional
        Whether to annotate the heatmap with correlation values

    Returns
    -------
    str
        Path to the saved heatmap
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create figure
    plt.figure(figsize=(14, 12))

    # Create mask for the upper triangle
    mask = np.zeros_like(correlation_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    # Set annotation format based on whether we have p-values
    if p_values is not None:
        # Create a custom annotation array with asterisks for significant correlations
        annot = np.empty_like(correlation_matrix, dtype=object)
        for i in range(correlation_matrix.shape[0]):
            for j in range(correlation_matrix.shape[1]):
                r = correlation_matrix.iloc[i, j]
                p = p_values.iloc[i, j]

                # Format the correlation value
                corr_str = f"{r:.2f}"

                # Add asterisks for significance
                if p < 0.001:
                    corr_str += "***"
                elif p < 0.01:
                    corr_str += "**"
                elif p < 0.05:
                    corr_str += "*"

                annot[i, j] = corr_str

        # Create heatmap with custom annotations
        sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, center=0,
                    annot=annot if annotate else False, fmt='', square=True, linewidths=.5,
                    cbar_kws={"shrink": .5})

        # Add legend for significance
        plt.annotate("* p<0.05, ** p<0.01, *** p<0.001", xy=(0.5, -0.05),
                     xycoords='axes fraction', ha='center', va='center', fontsize=10)
    else:
        # Create regular heatmap
        sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, center=0,
                    annot=annotate, fmt='.2f', square=True, linewidths=.5,
                    cbar_kws={"shrink": .5})

    # Set title and adjust layout
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()

    # Save and return path
    output_path = os.path.join(output_dir, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def create_correlation_network(correlation_matrix: pd.DataFrame,
                               p_values: pd.DataFrame = None,
                               min_correlation: float = 0.3,
                               max_links: int = 20,
                               title: str = "Correlation Network",
                               output_file: str = "correlation_network.png",
                               output_dir: str = 'results/phase3/figures') -> str:
    """
    Create and save a correlation network visualization showing the strongest relationships.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        Correlation matrix
    p_values : pd.DataFrame, optional
        P-values for the correlations
    min_correlation : float, optional
        Minimum absolute correlation to include in the network
    max_links : int, optional
        Maximum number of links to show in the network
    title : str, optional
        Title for the visualization
    output_file : str, optional
        Filename for the saved visualization
    output_dir : str, optional
        Directory to save the visualization

    Returns
    -------
    str
        Path to the saved visualization
    """
    try:
        import networkx as nx

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create figure
        plt.figure(figsize=(14, 12))

        # Create a new graph
        G = nx.Graph()

        # Add nodes
        for node in correlation_matrix.columns:
            G.add_node(node)

        # Extract correlations and filter by absolute value and significance
        correlations = []
        for i, col1 in enumerate(correlation_matrix.columns):
            for j, col2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicates and self-correlations
                    r = correlation_matrix.loc[col1, col2]
                    p = p_values.loc[col1, col2] if p_values is not None else 0

                    # Only keep correlations above threshold and significant if p-values provided
                    if abs(r) >= min_correlation and (p_values is None or p < 0.05):
                        correlations.append((col1, col2, abs(r), r, p))

        # Sort by absolute correlation and take the top max_links
        correlations = sorted(correlations, key=lambda x: x[2], reverse=True)[:max_links]

        # Add edges to the graph
        for col1, col2, abs_r, r, p in correlations:
            # Edge width proportional to correlation strength
            width = abs_r * 3

            # Edge color based on correlation sign (red for negative, blue for positive)
            color = 'red' if r < 0 else 'blue'

            G.add_edge(col1, col2, weight=width, color=color)

        # Get positions for nodes
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue', alpha=0.8)

        # Draw edges with varying width and color
        for u, v, attr in G.edges(data=True):
            width = attr['weight']
            color = attr['color']
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.7, edge_color=color)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

        # Add legend
        plt.plot([0], [0], linestyle='-', color='blue', linewidth=2, label='Positive Correlation')
        plt.plot([0], [0], linestyle='-', color='red', linewidth=2, label='Negative Correlation')
        plt.legend(loc='lower right')

        # Add title and remove axes
        plt.title(title, fontsize=16)
        plt.axis('off')

        # Save and return path
        output_path = os.path.join(output_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    except ImportError:
        logging.warning("NetworkX package not available, skipping correlation network visualization")
        return "NetworkX package required for correlation network visualization"


def create_scatter_matrix(df: pd.DataFrame,
                          variables: List[str],
                          color_var: str = 'target_binary',
                          title: str = "Scatter Matrix",
                          output_file: str = "scatter_matrix.png",
                          output_dir: str = 'results/phase3/figures') -> str:
    """
    Create and save a scatter matrix visualization for selected variables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    variables : List[str]
        List of variables to include in the scatter matrix
    color_var : str, optional
        Variable to use for color coding points
    title : str, optional
        Title for the visualization
    output_file : str, optional
        Filename for the saved visualization
    output_dir : str, optional
        Directory to save the visualization

    Returns
    -------
    str
        Path to the saved visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if variables exist
    variables = [var for var in variables if var in df.columns]
    if not variables:
        return "No valid variables found for scatter matrix"

    # Limit to 5 variables max to avoid overwhelming visualization
    if len(variables) > 5:
        variables = variables[:5]

    # Create figure
    sns.set(style="ticks")

    # Use color_var if it exists
    if color_var in df.columns:
        # Create a copy of the DataFrame with the selected variables
        plot_df = df[variables + [color_var]].copy()

        # For target_binary, use descriptive labels
        if color_var == 'target_binary':
            plot_df['Disease Status'] = plot_df[color_var].map({0: 'No Disease', 1: 'Disease'})
            color_var = 'Disease Status'

        # Create the scatter matrix
        g = sns.pairplot(plot_df, vars=variables, hue=color_var, diag_kind='kde',
                         plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'none'},
                         diag_kws={'alpha': 0.6}, height=2.5, aspect=1)
    else:
        # Create the scatter matrix without color
        g = sns.pairplot(df[variables], diag_kind='kde',
                         plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'none'},
                         diag_kws={'alpha': 0.6}, height=2.5, aspect=1)

    # Add correlations to the scatter plots
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        ax = g.axes[i, j]
        x = variables[j]
        y = variables[i]

        # Calculate Pearson correlation
        corr = df[[x, y]].corr().loc[x, y]

        # Add text with correlation value
        ax.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax.transAxes,
                ha='left', va='top', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7))

    # Set title
    g.fig.suptitle(title, y=1.02, fontsize=16)

    # Adjust layout
    g.fig.tight_layout()

    # Save and return path
    output_path = os.path.join(output_dir, output_file)
    g.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(g.fig)

    return output_path


def compare_correlations(pearson_corr: pd.DataFrame,
                         spearman_corr: pd.DataFrame,
                         min_difference: float = 0.15) -> List[Dict]:
    """
    Identify variable pairs with substantial differences between
    Pearson and Spearman correlations.

    Parameters
    ----------
    pearson_corr : pd.DataFrame
        Pearson correlation matrix
    spearman_corr : pd.DataFrame
        Spearman correlation matrix
    min_difference : float, optional
        Minimum absolute difference to consider (default: 0.15)

    Returns
    -------
    List[Dict]
        List of variable pairs with substantial differences
    """
    differences = []

    # Find common variables
    common_vars = [var for var in pearson_corr.columns if var in spearman_corr.columns]

    # Compare correlations
    for i, var1 in enumerate(common_vars):
        for j, var2 in enumerate(common_vars):
            if i < j:  # Only consider unique pairs
                pearson_val = pearson_corr.loc[var1, var2]
                spearman_val = spearman_corr.loc[var1, var2]
                diff = abs(pearson_val - spearman_val)

                if diff >= min_difference:
                    differences.append({
                        'variable1': var1,
                        'variable2': var2,
                        'pearson': pearson_val,
                        'spearman': spearman_val,
                        'difference': diff,
                        'likely_nonlinear': spearman_val > pearson_val
                    })

    # Sort by difference
    differences = sorted(differences, key=lambda x: x['difference'], reverse=True)

    return differences


def analyze_key_correlations(correlation_matrix: pd.DataFrame,
                             p_values: pd.DataFrame = None,
                             target_var: str = 'target_binary',
                             min_correlation: float = 0.2) -> List[Dict]:
    """
    Identify and analyze key correlations in the dataset.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        Correlation matrix
    p_values : pd.DataFrame, optional
        P-values for the correlations
    target_var : str, optional
        Target variable to focus on (default: 'target_binary')
    min_correlation : float, optional
        Minimum absolute correlation to consider (default: 0.2)

    Returns
    -------
    List[Dict]
        List of key correlations with analysis
    """
    correlations = []

    # Check if target variable exists in the matrix
    if target_var in correlation_matrix.columns:
        # Extract correlations with target variable
        for var in correlation_matrix.columns:
            if var != target_var:
                corr = correlation_matrix.loc[var, target_var]
                p = p_values.loc[var, target_var] if p_values is not None else None

                if abs(corr) >= min_correlation:
                    entry = {
                        'var1': var,
                        'var2': target_var,
                        'value': corr,
                        'abs_value': abs(corr),
                        'p_value': p,
                        'significant': p < 0.05 if p is not None else None,
                        'direction': 'positive' if corr > 0 else 'negative'
                    }
                    correlations.append(entry)

    # Extract other strong correlations
    for i, var1 in enumerate(correlation_matrix.columns):
        for j, var2 in enumerate(correlation_matrix.columns):
            if i < j and var1 != target_var and var2 != target_var:  # Unique pairs excluding target
                corr = correlation_matrix.loc[var1, var2]
                p = p_values.loc[var1, var2] if p_values is not None else None

                if abs(corr) >= min_correlation:
                    entry = {
                        'var1': var1,
                        'var2': var2,
                        'value': corr,
                        'abs_value': abs(corr),
                        'p_value': p,
                        'significant': p < 0.05 if p is not None else None,
                        'direction': 'positive' if corr > 0 else 'negative'
                    }
                    correlations.append(entry)

    # Sort by absolute correlation value
    correlations = sorted(correlations, key=lambda x: x['abs_value'], reverse=True)

    return correlations


def create_correlation_barplot(df: pd.DataFrame,
                               target_var: str,
                               top_n: int = 10,
                               corr_type: str = 'pearson',
                               title: str = "Top Correlations with Target Variable",
                               output_file: str = "correlation_barplot.png",
                               output_dir: str = 'results/phase3/figures') -> str:
    """
    Create and save a barplot of the top correlations with the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    target_var : str
        Target variable
    top_n : int, optional
        Number of top correlations to show (default: 10)
    corr_type : str, optional
        Type of correlation to calculate (default: 'pearson')
    title : str, optional
        Title for the visualization
    output_file : str, optional
        Filename for the saved visualization
    output_dir : str, optional
        Directory to save the visualization

    Returns
    -------
    str
        Path to the saved visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if target variable exists
    if target_var not in df.columns:
        return f"Target variable '{target_var}' not found"

    # Calculate correlations with target variable
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_var
                    and not col.endswith('_missing') and not col.endswith('_outlier')
                    and not col.endswith('_original') and col != 'source']

    # Calculate correlations and p-values
    correlations = []
    for col in numeric_cols:
        if corr_type == 'pearson':
            corr, p = stats.pearsonr(df[col].dropna(), df[target_var].dropna())
        else:  # spearman
            corr, p = stats.spearmanr(df[col].dropna(), df[target_var].dropna())

        correlations.append({
            'variable': col,
            'correlation': corr,
            'p_value': p,
            'significant': p < 0.05
        })

    # Sort by absolute correlation
    correlations = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)

    # Take top N
    top_correlations = correlations[:top_n]

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(top_correlations)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Create barplot
    bars = plt.barh(plot_df['variable'], plot_df['correlation'],
                    color=[plt.cm.RdBu(0.1) if x < 0 else plt.cm.RdBu(0.9) for x in plot_df['correlation']])

    # Add significance markers
    for i, corr in enumerate(top_correlations):
        if corr['significant']:
            plt.text(corr['correlation'] * 1.05 if corr['correlation'] > 0 else corr['correlation'] * 0.95,
                     i, '*', fontsize=14, ha='center', va='center')

    # Add correlation values
    for i, corr in enumerate(top_correlations):
        plt.text(corr['correlation'] * 0.9 if corr['correlation'] > 0 else corr['correlation'] * 1.1,
                 i, f"{corr['correlation']:.2f}", fontsize=10,
                 ha='right' if corr['correlation'] > 0 else 'left',
                 va='center', color='white' if abs(corr['correlation']) > 0.4 else 'black')

    # Add labels and title
    plt.xlabel(f'{corr_type.capitalize()} Correlation Coefficient')
    plt.title(title)

    # Add zero line
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # Add legend for significance
    plt.annotate("* p < 0.05", xy=(0.95, 0.02), xycoords='axes fraction',
                 ha='right', va='bottom', fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save and return path
    output_path = os.path.join(output_dir, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def perform_correlation_analysis(df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive correlation analysis on the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset for analysis

    Returns
    -------
    Dict
        Dictionary containing all correlation analysis results
    """
    results = {}

    print("Performing correlation analysis...")

    # Define variables to include in the analysis
    key_variables = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
                     'num', 'target_binary']

    # Keep only variables that exist in the dataset
    key_variables = [var for var in key_variables if var in df.columns]

    # 1. Calculate Pearson correlations
    print("\n1. Calculating Pearson correlations...")
    pearson_results = calculate_pearson_correlations(df, key_variables)
    results['pearson_correlations'] = pearson_results

    # Convert dictionary results to DataFrame for easier manipulation
    pearson_matrix = pd.DataFrame(pearson_results['correlation_matrix'])
    pearson_p_values = pd.DataFrame(pearson_results['p_values']) if 'p_values' in pearson_results else None

    # Create correlation heatmap
    pearson_heatmap = create_correlation_heatmap(
        pearson_matrix,
        pearson_p_values,
        title="Pearson Correlation Heatmap",
        output_file="pearson_correlation_heatmap.png"
    )
    results['pearson_heatmap'] = pearson_heatmap

    # 2. Calculate Spearman correlations
    print("\n2. Calculating Spearman correlations...")
    spearman_results = calculate_spearman_correlations(df, key_variables)
    results['spearman_correlations'] = spearman_results

    # Convert dictionary results to DataFrame for easier manipulation
    spearman_matrix = pd.DataFrame(spearman_results['correlation_matrix'])
    spearman_p_values = pd.DataFrame(spearman_results['p_values']) if 'p_values' in spearman_results else None

    # Create correlation heatmap
    spearman_heatmap = create_correlation_heatmap(
        spearman_matrix,
        spearman_p_values,
        title="Spearman Correlation Heatmap",
        output_file="spearman_correlation_heatmap.png"
    )
    results['spearman_heatmap'] = spearman_heatmap

    # 3. Calculate partial correlations
    print("\n3. Calculating partial correlations...")
    partial_results = calculate_partial_correlations(df, key_variables, ['age', 'sex'])
    results['partial_correlations'] = partial_results

    # Create partial correlation heatmap if calculation was successful
    if 'error' not in partial_results:
        partial_matrix = pd.DataFrame(partial_results['correlation_matrix'])
        partial_p_values = pd.DataFrame(partial_results['p_values']) if 'p_values' in partial_results else None

        partial_heatmap = create_correlation_heatmap(
            partial_matrix,
            partial_p_values,
            title=f"Partial Correlation Heatmap (Controlling for {', '.join(partial_results['control_variables'])})",
            output_file="partial_correlation_heatmap.png"
        )
        results['partial_heatmap'] = partial_heatmap

    # 4. Create correlation barplots for target variable
    if 'target_binary' in df.columns:
        print("\n4. Creating correlation barplots for target variable...")
        pearson_barplot = create_correlation_barplot(
            df,
            'target_binary',
            corr_type='pearson',
            title="Top Pearson Correlations with Heart Disease",
            output_file="pearson_target_correlations.png"
        )
        results['pearson_target_barplot'] = pearson_barplot

        spearman_barplot = create_correlation_barplot(
            df,
            'target_binary',
            corr_type='spearman',
            title="Top Spearman Correlations with Heart Disease",
            output_file="spearman_target_correlations.png"
        )
        results['spearman_target_barplot'] = spearman_barplot

    # 5. Create scatter matrix for key predictors
    if 'target_binary' in df.columns:
        print("\n5. Creating scatter matrix for key predictors...")
        # Identify key predictors from Pearson correlations
        key_predictors = []
        if pearson_matrix is not None and 'target_binary' in pearson_matrix.columns:
            # Get correlations with target and sort by absolute value
            target_corrs = pearson_matrix['target_binary'].drop('target_binary')
            key_predictors = target_corrs.abs().sort_values(ascending=False).index.tolist()[:5]

        if key_predictors:
            scatter_plot = create_scatter_matrix(
                df,
                key_predictors,
                color_var='target_binary',
                title="Scatter Matrix of Key Predictors",
                output_file="key_predictors_scatter_matrix.png"
            )
            results['key_predictors_scatter'] = scatter_plot

    # 6. Compare Pearson and Spearman correlations
    print("\n6. Comparing Pearson and Spearman correlations...")
    if pearson_matrix is not None and spearman_matrix is not None:
        correlation_differences = compare_correlations(pearson_matrix, spearman_matrix)
        results['correlation_differences'] = correlation_differences

    # 7. Analyze key correlations
    print("\n7. Analyzing key correlations...")
    key_correlations = []
    if pearson_matrix is not None and pearson_p_values is not None:
        key_correlations = analyze_key_correlations(pearson_matrix, pearson_p_values)
    results['key_correlations'] = key_correlations

    # 8. Identify partial correlation insights
    print("\n8. Identifying partial correlation insights...")
    partial_insights = []

    if 'error' not in partial_results and pearson_matrix is not None:
        partial_matrix = pd.DataFrame(partial_results['correlation_matrix'])

        # Compare partial correlations with Pearson correlations
        for i, var1 in enumerate(partial_matrix.columns):
            for j, var2 in enumerate(partial_matrix.columns):
                if i < j:  # Only unique pairs
                    pearson_val = pearson_matrix.loc[var1, var2]
                    partial_val = partial_matrix.loc[var1, var2]
                    diff = abs(pearson_val - partial_val)

                    # If significant difference, add to insights
                    if diff > 0.1:
                        if abs(partial_val) < abs(pearson_val):
                            # Correlation was reduced after controlling
                            insight = f"The correlation between {var1} and {var2} decreases from {pearson_val:.2f} to {partial_val:.2f} after controlling for {', '.join(partial_results['control_variables'])}, suggesting a confounding effect."
                        else:
                            # Correlation was increased or switched direction after controlling
                            insight = f"The correlation between {var1} and {var2} changes from {pearson_val:.2f} to {partial_val:.2f} after controlling for {', '.join(partial_results['control_variables'])}, suggesting a suppression effect."

                        partial_insights.append(insight)

    results['partial_correlation_insights'] = partial_insights

    # 9. Generate clinical interpretations based on correlations
    print("\n9. Generating clinical interpretations...")
    clinical_interpretations = []

    # Interpretations from key correlations
    if 'target_binary' in df.columns and pearson_matrix is not None:
        target_corrs = {}
        if 'target_binary' in pearson_matrix.columns:
            for var in pearson_matrix.columns:
                if var != 'target_binary':
                    target_corrs[var] = pearson_matrix.loc[var, 'target_binary']

        # Interpret strong correlations with target
        for var, corr in target_corrs.items():
            if abs(corr) >= 0.3:
                if var == 'age' and corr > 0:
                    clinical_interpretations.append(
                        f"Age shows a positive correlation (r={corr:.2f}) with heart disease, confirming that cardiac risk increases with age.")
                elif var == 'sex' and corr != 0:
                    direction = "higher" if corr > 0 else "lower"
                    clinical_interpretations.append(
                        f"Gender correlates with heart disease (r={corr:.2f}), with males showing {direction} risk in this dataset.")
                elif var == 'chol' and corr > 0:
                    clinical_interpretations.append(
                        f"Cholesterol shows a positive correlation (r={corr:.2f}) with heart disease, supporting its role as a modifiable risk factor.")
                elif var == 'thalach' and corr < 0:
                    clinical_interpretations.append(
                        f"Maximum heart rate achieved shows a negative correlation (r={corr:.2f}) with heart disease, suggesting reduced cardiac functional capacity in disease patients.")
                elif var == 'oldpeak' and corr > 0:
                    clinical_interpretations.append(
                        f"ST depression (oldpeak) strongly correlates (r={corr:.2f}) with heart disease, confirming its value as a diagnostic indicator on ECG.")
                elif var == 'cp' and corr != 0:
                    clinical_interpretations.append(
                        f"Chest pain type shows a significant correlation (r={corr:.2f}) with heart disease, highlighting the importance of pain characteristics in diagnosis.")
                elif abs(corr) >= 0.4:  # Very strong correlations
                    direction = "positive" if corr > 0 else "negative"
                    clinical_interpretations.append(
                        f"{var} shows a strong {direction} correlation (r={corr:.2f}) with heart disease, suggesting its potential importance in risk assessment.")

    # Interpretations from partial correlations
    if partial_insights:
        # Select the most interesting insights
        top_insights = partial_insights[:3]
        for insight in top_insights:
            clinical_interpretations.append(insight)

    # Interpretations from correlation differences
    if 'correlation_differences' in results and results['correlation_differences']:
        top_diffs = results['correlation_differences'][:2]
        for diff in top_diffs:
            if diff['likely_nonlinear']:
                clinical_interpretations.append(
                    f"The relationship between {diff['variable1']} and {diff['variable2']} appears non-linear, as Spearman correlation (r={diff['spearman']:.2f}) is stronger than Pearson (r={diff['pearson']:.2f}).")

    results['clinical_interpretations'] = clinical_interpretations

    print("\nCorrelation analysis complete.")

    return results


if __name__ == "__main__":
    # Test the module if run directly
    print("Correlation Analysis Module for the Heart Disease dataset.")
    print("This module provides correlation analysis functions for the Heart Disease dataset.")
    print("Import and use in another script or notebook.")