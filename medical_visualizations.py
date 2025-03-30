"""
Medical Visualization Module for Heart Disease Dataset

This module implements specialized medical visualizations for the UCI
Heart Disease dataset, focusing on clinical indicators and subgroup comparisons.

Author: Team Member 1
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


def setup_viz_environment():
    """Set up the visualization environment with customized settings for medical plots."""
    # Set seaborn style for medical visualization
    sns.set_style("whitegrid")

    # Medical-themed color palette (blues, greens, reds for medical significance)
    colors = ["#2C7FB8", "#7FCDBB", "#D95F0E", "#FEC44F", "#FFFFCC", "#FC8D59"]
    sns.set_palette(sns.color_palette(colors))

    # Set matplotlib parameters for consistent medical visualizations
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    # Title style that looks more like medical publications
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['figure.titleweight'] = 'bold'


def create_advanced_pairplot(df: pd.DataFrame, output_dir: str) -> str:
    """
    Create an enhanced pairplot showing relationships between key clinical indicators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the variables
    output_dir : str
        Directory to save the visualization

    Returns
    -------
    str
        Path to the saved visualization
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Select key clinical variables
    clinical_vars = ['age', 'thalach', 'oldpeak', 'chol', 'trestbps']

    # Prepare data
    plot_df = df.copy()

    # Add disease status with proper labels
    if 'target_binary' in df.columns:
        plot_df['Disease Status'] = plot_df['target_binary'].map({0: 'No Disease', 1: 'Disease'})
        hue_var = 'Disease Status'
    else:
        hue_var = None

    # Create enhanced pairplot
    g = sns.pairplot(
        plot_df,
        vars=clinical_vars,
        hue=hue_var,
        diag_kind='kde',
        corner=True,  # Show only lower triangle
        plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'w', 'linewidth': 0.5},
        diag_kws={'alpha': 0.6, 'fill': True, 'common_norm': False},
        height=2.5,
        aspect=1.2
    )

    # Add correlation coefficients to the plots
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        # Skip upper triangle
        continue

    for i, j in zip(*np.tril_indices_from(g.axes, -1)):
        ax = g.axes[i, j]
        x = clinical_vars[j]
        y = clinical_vars[i]

        # Add correlation for the entire dataset
        corr = plot_df[[x, y]].corr().iloc[0, 1]

        # Format correlation text
        corr_text = f"r = {corr:.2f}"

        # Add correlation text to the plot
        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes,
                ha='left', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray'))

        # If we have disease status, add correlations by group
        if hue_var:
            for status, color in zip(['No Disease', 'Disease'], g._legend.get_lines()):
                group_df = plot_df[plot_df[hue_var] == status]
                group_corr = group_df[[x, y]].corr().iloc[0, 1]

                # Get color from legend
                c = color.get_color()

                # Add correlation text by group at bottom
                if status == 'No Disease':
                    ax.text(0.05, 0.05, f"{status}: r = {group_corr:.2f}",
                            transform=ax.transAxes, ha='left', va='bottom',
                            fontsize=8, color=c,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                else:
                    ax.text(0.95, 0.05, f"{status}: r = {group_corr:.2f}",
                            transform=ax.transAxes, ha='right', va='bottom',
                            fontsize=8, color=c,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Customize the title
    g.fig.suptitle('Relationships Between Key Clinical Indicators', y=1.02)

    # Add clinical interpretations as figure text
    interpretations = [
        "• Age shows inverse relationship with max heart rate",
        "• ST depression (oldpeak) higher in disease patients",
        "• Cholesterol shows complex relationship with other indicators",
        "• Resting blood pressure weakly associated with other parameters"
    ]

    interpretation_text = "\n".join(interpretations)
    g.fig.text(0.02, 0.02, "Clinical Interpretations:", fontweight='bold', fontsize=12)
    g.fig.text(0.02, -0.02, interpretation_text, fontsize=10)

    # Adjust layout
    plt.tight_layout()
    g.fig.subplots_adjust(bottom=0.15)

    # Save the figure
    output_path = os.path.join(output_dir, 'clinical_indicators_pairplot.png')
    g.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(g.fig)

    logging.info(f"Advanced pairplot created and saved to {output_path}")
    return output_path


def create_faceted_plots(df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    """
    Create faceted plots comparing distributions across patient subgroups.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the variables
    output_dir : str
        Directory to save the visualizations

    Returns
    -------
    Dict[str, str]
        Dictionary mapping plot types to their file paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_files = {}

    # 1. Age and Chest Pain Type Faceted by Gender
    fig = plt.figure(figsize=(14, 10))

    # Create a copy with mapped categories for better display
    plot_df = df.copy()

    # Map categorical variables
    if 'cp' in plot_df.columns:
        plot_df['Chest Pain Type'] = plot_df['cp'].map({
            1: 'Typical Angina',
            2: 'Atypical Angina',
            3: 'Non-anginal Pain',
            4: 'Asymptomatic'
        })

    if 'sex' in plot_df.columns:
        plot_df['Gender'] = plot_df['sex'].map({0: 'Female', 1: 'Male'})

    if 'target_binary' in plot_df.columns:
        plot_df['Disease Status'] = plot_df['target_binary'].map({0: 'No Disease', 1: 'Disease'})

    # Create FacetGrid
    if all(col in plot_df.columns for col in ['Chest Pain Type', 'Gender', 'Disease Status']):
        g = sns.FacetGrid(
            plot_df,
            col='Chest Pain Type',
            row='Gender',
            hue='Disease Status',
            height=4,
            aspect=1.2,
            margin_titles=True
        )

        # Map boxplot of age
        g.map_dataframe(sns.boxplot, x='Disease Status', y='age')

        # Add count annotations
        def annotate_counts(data, **kws):
            ax = plt.gca()
            for i, status in enumerate(data['Disease Status'].unique()):
                count = len(data[data['Disease Status'] == status])
                ax.text(i, data['age'].min() - 5, f"n={count}",
                        ha='center', va='top', fontsize=9)

        g.map_dataframe(annotate_counts)

        # Finalize the grid
        g.set_axis_labels('', 'Age (years)')
        g.set_titles(col_template='{col_name}', row_template='{row_name}')
        g.add_legend(title='Disease Status')
        g.fig.suptitle('Age Distribution by Gender, Chest Pain Type, and Disease Status', y=1.02)

        # Add clinical interpretation
        g.fig.text(0.5, 0.01,
                   "Clinical Note: Asymptomatic chest pain shows highest disease prevalence across both genders. "
                   "Males with asymptomatic chest pain show higher median age with disease.",
                   ha='center', fontsize=11, style='italic',
                   bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))

        # Save the figure
        output_path = os.path.join(output_dir, 'age_by_gender_cp_facet.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        output_files['age_gender_cp_facet'] = output_path
        logging.info(f"Gender-CP faceted plot created and saved to {output_path}")

    # 2. Create cholesterol and thalach faceted by age group and disease status
    if 'age' in plot_df.columns:
        # Create age groups if they don't exist
        if 'age_group' not in plot_df.columns:
            bins = [0, 40, 50, 60, 70, 100]
            labels = ['<40', '40-49', '50-59', '60-69', '70+']
            plot_df['Age Group'] = pd.cut(plot_df['age'], bins=bins, labels=labels)
        else:
            plot_df['Age Group'] = plot_df['age_group']

    if all(col in plot_df.columns for col in ['Age Group', 'Disease Status']):
        # Cholesterol plot
        plt.figure(figsize=(14, 10))
        g_chol = sns.FacetGrid(
            plot_df,
            col='Age Group',
            row='Disease Status',
            height=4,
            aspect=1.2,
            margin_titles=True
        )

        # Map violinplot and stripplot
        g_chol.map_dataframe(sns.violinplot, x='Gender', y='chol', palette='Set2', cut=0)
        g_chol.map_dataframe(sns.stripplot, x='Gender', y='chol', color='black', alpha=0.3, jitter=True, size=3)

        # Add statistics
        def add_stats(data, **kws):
            ax = plt.gca()
            if len(data) > 1:
                # Calculate mean and median
                mean = data['chol'].mean()
                median = data['chol'].median()

                # Add statistics text
                stats_text = f"Mean: {mean:.1f}\nMedian: {median:.1f}"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                        ha='right', va='top', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

                # If we have both genders, add t-test
                if len(data['Gender'].unique()) > 1:
                    female_data = data[data['Gender'] == 'Female']['chol']
                    male_data = data[data['Gender'] == 'Male']['chol']

                    if len(female_data) > 0 and len(male_data) > 0:
                        _, p_val = stats.ttest_ind(female_data, male_data, equal_var=False)

                        # Add p-value annotation
                        sig_text = f"p = {p_val:.3f}"
                        if p_val < 0.05:
                            sig_text += " *"
                        ax.text(0.5, 0.04, sig_text, transform=ax.transAxes,
                                ha='center', va='bottom', fontsize=9)

        g_chol.map_dataframe(add_stats)

        # Finalize the grid
        g_chol.set_axis_labels('Gender', 'Cholesterol (mg/dl)')
        g_chol.set_titles(col_template='{col_name}', row_template='{row_name}')
        g_chol.fig.suptitle('Cholesterol Distribution by Age Group, Disease Status, and Gender', y=1.02)

        # Add clinical interpretation
        g_chol.fig.text(0.5, 0.01,
                        "Clinical Note: Gender differences in cholesterol are most pronounced in middle age groups. "
                        "Disease status affects cholesterol distribution patterns across age groups.",
                        ha='center', fontsize=11, style='italic',
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))

        # Save the figure
        output_path = os.path.join(output_dir, 'cholesterol_facet.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        output_files['cholesterol_facet'] = output_path
        logging.info(f"Cholesterol faceted plot created and saved to {output_path}")

        # Heart rate plot (thalach)
        plt.figure(figsize=(14, 10))
        g_hr = sns.FacetGrid(
            plot_df,
            col='Age Group',
            row='Disease Status',
            height=4,
            aspect=1.2,
            margin_titles=True
        )

        # Map violinplot and swarmplot
        g_hr.map_dataframe(sns.violinplot, x='Gender', y='thalach', palette='Set2', cut=0)
        g_hr.map_dataframe(sns.stripplot, x='Gender', y='thalach', color='black', alpha=0.3, jitter=True, size=3)

        # Add statistics (reusing function defined above)
        def add_thalach_stats(data, **kws):
            ax = plt.gca()
            if len(data) > 1:
                # Calculate mean and median
                mean = data['thalach'].mean()
                median = data['thalach'].median()

                # Add statistics text
                stats_text = f"Mean: {mean:.1f}\nMedian: {median:.1f}"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                        ha='right', va='top', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

                # If we have both genders, add t-test
                if len(data['Gender'].unique()) > 1:
                    female_data = data[data['Gender'] == 'Female']['thalach']
                    male_data = data[data['Gender'] == 'Male']['thalach']

                    if len(female_data) > 0 and len(male_data) > 0:
                        _, p_val = stats.ttest_ind(female_data, male_data, equal_var=False)

                        # Add p-value annotation
                        sig_text = f"p = {p_val:.3f}"
                        if p_val < 0.05:
                            sig_text += " *"
                        ax.text(0.5, 0.04, sig_text, transform=ax.transAxes,
                                ha='center', va='bottom', fontsize=9)

        g_hr.map_dataframe(add_thalach_stats)

        # Finalize the grid
        g_hr.set_axis_labels('Gender', 'Maximum Heart Rate (bpm)')
        g_hr.set_titles(col_template='{col_name}', row_template='{row_name}')
        g_hr.fig.suptitle('Maximum Heart Rate Distribution by Age Group, Disease Status, and Gender', y=1.02)

        # Add clinical interpretation
        g_hr.fig.text(0.5, 0.01,
                      "Clinical Note: Maximum heart rate consistently lower in disease patients across age groups. "
                      "Age-related decrease in maximum heart rate more pronounced than gender differences.",
                      ha='center', fontsize=11, style='italic',
                      bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))

        # Save the figure
        output_path = os.path.join(output_dir, 'heart_rate_facet.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        output_files['heart_rate_facet'] = output_path
        logging.info(f"Heart rate faceted plot created and saved to {output_path}")

    return output_files


def create_violin_plots(df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    """
    Create enhanced violin plots showing distribution shapes and key statistics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the variables
    output_dir : str
        Directory to save the visualizations

    Returns
    -------
    Dict[str, str]
        Dictionary mapping plot types to their file paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_files = {}

    # Create a copy with mapped categories for better display
    plot_df = df.copy()

    # Map categorical variables
    if 'sex' in plot_df.columns:
        plot_df['Gender'] = plot_df['sex'].map({0: 'Female', 1: 'Male'})

    if 'cp' in plot_df.columns:
        plot_df['Chest Pain Type'] = plot_df['cp'].map({
            1: 'Typical Angina',
            2: 'Atypical Angina',
            3: 'Non-anginal Pain',
            4: 'Asymptomatic'
        })

    if 'target_binary' in plot_df.columns:
        plot_df['Disease Status'] = plot_df['target_binary'].map({0: 'No Disease', 1: 'Disease'})

    # 1. ST Depression (oldpeak) Enhanced Violin Plot
    if all(col in plot_df.columns for col in ['Disease Status', 'oldpeak']):
        plt.figure(figsize=(14, 8))

        # Create a figure with subplots: one for the violin plot, one for density curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [2, 1]})

        # Enhanced violin plot with individual points and box plot inside
        sns.violinplot(x='Disease Status', y='oldpeak', data=plot_df,
                       ax=ax1, inner='box', palette=['#6baed6', '#fd8d3c'])

        # Add individual points with jitter
        sns.stripplot(x='Disease Status', y='oldpeak', data=plot_df,
                      ax=ax1, color='black', alpha=0.3, jitter=True, size=3)

        # Add statistical annotation
        no_disease = plot_df[plot_df['Disease Status'] == 'No Disease']['oldpeak']
        disease = plot_df[plot_df['Disease Status'] == 'Disease']['oldpeak']

        # T-test
        t_stat, p_val = stats.ttest_ind(no_disease, disease, equal_var=False)

        # Format p-value
        if p_val < 0.001:
            p_text = "p < 0.001"
        else:
            p_text = f"p = {p_val:.3f}"

        # Calculate effect size (Cohen's d)
        mean_diff = disease.mean() - no_disease.mean()
        pooled_std = np.sqrt(((len(disease) - 1) * disease.std() ** 2 +
                              (len(no_disease) - 1) * no_disease.std() ** 2) /
                             (len(disease) + len(no_disease) - 2))
        cohens_d = abs(mean_diff) / pooled_std

        # Effect size interpretation
        if cohens_d < 0.5:
            effect_text = "Small Effect"
        elif cohens_d < 0.8:
            effect_text = "Medium Effect"
        else:
            effect_text = "Large Effect"

        # Add statistical annotation
        stats_text = (f"T-test: {t_stat:.2f}, {p_text}\n"
                      f"Cohen's d: {cohens_d:.2f} ({effect_text})\n"
                      f"Mean Difference: {mean_diff:.2f}")

        ax1.text(0.5, 0.02, stats_text, transform=ax1.transAxes,
                 ha='center', va='bottom', fontsize=11,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

        # Add density curves
        sns.kdeplot(data=no_disease, ax=ax2, color='#6baed6', fill=True,
                    common_norm=False, alpha=0.7, linewidth=2, label='No Disease')
        sns.kdeplot(data=disease, ax=ax2, color='#fd8d3c', fill=True,
                    common_norm=False, alpha=0.7, linewidth=2, label='Disease')

        # Add mean lines to density plot
        ax2.axvline(no_disease.mean(), color='#6baed6', linestyle='--', linewidth=2,
                    label=f'No Disease Mean: {no_disease.mean():.2f}')
        ax2.axvline(disease.mean(), color='#fd8d3c', linestyle='--', linewidth=2,
                    label=f'Disease Mean: {disease.mean():.2f}')

        # Set labels and titles
        ax1.set_title('ST Depression (oldpeak) Distribution by Disease Status')
        ax1.set_xlabel('Disease Status')
        ax1.set_ylabel('ST Depression (mm)')

        ax2.set_title('Density Distribution')
        ax2.set_xlabel('ST Depression (mm)')
        ax2.set_ylabel('Density')
        ax2.legend(loc='upper right')

        # Add clinical interpretation
        fig.text(0.5, 0.01,
                 "Clinical Interpretation: ST depression during exercise is significantly higher in patients with heart disease. "
                 "This indicates ischemic changes during stress testing are a strong diagnostic marker.",
                 ha='center', fontsize=11, style='italic',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        # Save the figure
        output_path = os.path.join(output_dir, 'st_depression_violin.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        output_files['st_depression_violin'] = output_path
        logging.info(f"ST depression violin plot created and saved to {output_path}")

    # 2. Maximum Heart Rate by Chest Pain Type and Disease Status
    if all(col in plot_df.columns for col in ['Chest Pain Type', 'Disease Status', 'thalach']):
        plt.figure(figsize=(16, 10))

        # Create violin plot
        ax = sns.violinplot(x='Chest Pain Type', y='thalach', hue='Disease Status',
                            data=plot_df, split=True, inner='quartile',
                            palette=['#6baed6', '#fd8d3c'])

        # Add individual points with smaller size
        sns.stripplot(x='Chest Pain Type', y='thalach', hue='Disease Status',
                      data=plot_df, dodge=True, alpha=0.3,
                      palette=['#08519c', '#a63603'], size=3)

        # Calculate statistics for each chest pain type
        cp_types = plot_df['Chest Pain Type'].unique()

        # Add annotations for each chest pain type
        for i, cp in enumerate(sorted(cp_types)):
            # Get data for this chest pain type
            cp_data = plot_df[plot_df['Chest Pain Type'] == cp]
            no_disease = cp_data[cp_data['Disease Status'] == 'No Disease']['thalach']
            disease = cp_data[cp_data['Disease Status'] == 'Disease']['thalach']

            # Only add statistics if we have enough data in both groups
            if len(no_disease) > 0 and len(disease) > 0:
                # T-test
                t_stat, p_val = stats.ttest_ind(no_disease, disease, equal_var=False)

                # Format p-value
                if p_val < 0.001:
                    p_text = "p<0.001"
                else:
                    p_text = f"p={p_val:.3f}"

                # Add annotation
                plt.annotate(p_text, xy=(i, min(no_disease.min(), disease.min()) - 5),
                             ha='center', va='top', fontsize=10)

        # Add title and labels
        plt.title('Maximum Heart Rate by Chest Pain Type and Disease Status', fontsize=16)
        plt.xlabel('Chest Pain Type', fontsize=14)
        plt.ylabel('Maximum Heart Rate (bpm)', fontsize=14)

        # Fix legend (remove duplicate entries)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:2], labels[:2], title='Disease Status')

        # Add clinical interpretation
        plt.figtext(0.5, 0.01,
                    "Clinical Interpretation: Patients with heart disease consistently show lower maximum heart rates across all chest pain types. "
                    "This suggests reduced cardiac functional capacity in disease patients regardless of presenting symptoms.",
                    ha='center', fontsize=11, style='italic',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        # Save the figure
        output_path = os.path.join(output_dir, 'heart_rate_cp_violin.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        output_files['heart_rate_cp_violin'] = output_path
        logging.info(f"Heart rate by CP violin plot created and saved to {output_path}")

    # 3. Age by Major Vessels (ca) and Disease Status
    if all(col in plot_df.columns for col in ['ca', 'Disease Status', 'age']):
        plt.figure(figsize=(14, 8))

        # Map ca to readable labels
        plot_df['Major Vessels'] = plot_df['ca'].astype(str) + ' vessels'

        # Create violin plot
        ax = sns.violinplot(x='Major Vessels', y='age', hue='Disease Status',
                            data=plot_df, split=True, inner='quartile',
                            palette=['#6baed6', '#fd8d3c'])

        # Add individual points
        sns.stripplot(x='Major Vessels', y='age', hue='Disease Status',
                      data=plot_df, dodge=True, alpha=0.3,
                      palette=['#08519c', '#a63603'], size=3)

        # Add sample size annotations
        for i, vessels in enumerate(sorted(plot_df['Major Vessels'].unique())):
            subset = plot_df[plot_df['Major Vessels'] == vessels]
            count_no_disease = len(subset[subset['Disease Status'] == 'No Disease'])
            count_disease = len(subset[subset['Disease Status'] == 'Disease'])

            # Add sample size text
            plt.annotate(f"n={count_no_disease}", xy=(i - 0.2, plot_df['age'].min() - 2),
                         ha='center', va='top', fontsize=9, color='#08519c')
            plt.annotate(f"n={count_disease}", xy=(i + 0.2, plot_df['age'].min() - 2),
                         ha='center', va='top', fontsize=9, color='#a63603')

        # Add title and labels
        plt.title('Age Distribution by Number of Major Vessels Colored and Disease Status', fontsize=16)
        plt.xlabel('Number of Major Vessels Colored by Fluoroscopy', fontsize=14)
        plt.ylabel('Age (years)', fontsize=14)

        # Fix legend (remove duplicate entries)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:2], labels[:2], title='Disease Status')

        # Add clinical interpretation
        plt.figtext(0.5, 0.01,
                    "Clinical Interpretation: Number of major vessels colored by fluoroscopy strongly correlates with disease status. "
                    "Age distribution varies across vessel groups, with older patients showing more affected vessels.",
                    ha='center', fontsize=11, style='italic',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        # Save the figure
        output_path = os.path.join(output_dir, 'age_vessels_violin.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        output_files['age_vessels_violin'] = output_path
        logging.info(f"Age by vessels violin plot created and saved to {output_path}")

    return output_files


def create_clinical_risk_visualization(df: pd.DataFrame, output_dir: str) -> str:
    """
    Create a clinical risk factor visualization showing the impact of multiple factors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the variables
    output_dir : str
        Directory to save the visualization

    Returns
    -------
    str
        Path to the saved visualization
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a copy with mapped categories for better display
    plot_df = df.copy()

    # Define risk factors
    if 'target_binary' not in plot_df.columns:
        logging.warning("Target variable not found, cannot create risk visualization")
        return ""

    # Map categorical variables
    if 'sex' in plot_df.columns:
        plot_df['Gender'] = plot_df['sex'].map({0: 'Female', 1: 'Male'})

    if 'cp' in plot_df.columns:
        plot_df['Chest Pain Type'] = plot_df['cp'].map({
            1: 'Typical Angina',
            2: 'Atypical Angina',
            3: 'Non-anginal Pain',
            4: 'Asymptomatic'
        })

    if 'thal' in plot_df.columns:
        plot_df['Thalassemia'] = plot_df['thal'].map({
            3: 'Normal',
            6: 'Fixed Defect',
            7: 'Reversible Defect'
        })

    # Create risk factors
    risk_factors = []

    # Check if we have necessary columns for each risk factor
    if 'Gender' in plot_df.columns:
        plot_df['Male Gender'] = (plot_df['Gender'] == 'Male').astype(int)
        risk_factors.append('Male Gender')

    if 'age' in plot_df.columns:
        plot_df['Age > 55'] = (plot_df['age'] > 55).astype(int)
        risk_factors.append('Age > 55')

    if 'Chest Pain Type' in plot_df.columns:
        plot_df['Asymptomatic CP'] = (plot_df['Chest Pain Type'] == 'Asymptomatic').astype(int)
        risk_factors.append('Asymptomatic CP')

    if 'thalach' in plot_df.columns:
        plot_df['Low Max HR'] = (plot_df['thalach'] < 150).astype(int)
        risk_factors.append('Low Max HR')

    if 'oldpeak' in plot_df.columns:
        plot_df['ST Depression > 1'] = (plot_df['oldpeak'] > 1).astype(int)
        risk_factors.append('ST Depression > 1')

    if 'ca' in plot_df.columns:
        plot_df['Vessels > 0'] = (plot_df['ca'] > 0).astype(int)
        risk_factors.append('Vessels > 0')

    if 'Thalassemia' in plot_df.columns:
        plot_df['Reversible Defect'] = (plot_df['Thalassemia'] == 'Reversible Defect').astype(int)
        risk_factors.append('Reversible Defect')

    # Calculate odds ratios for each risk factor
    odds_ratios = {}
    confidence_intervals = {}
    p_values = {}

    for factor in risk_factors:
        # Create contingency table
        contingency = pd.crosstab(plot_df[factor], plot_df['target_binary'])

        # Ensure we have complete data
        if contingency.shape == (2, 2) and 0 not in contingency.values:
            # Calculate odds ratio
            a = contingency.iloc[1, 1]  # Factor present, Disease present
            b = contingency.iloc[1, 0]  # Factor present, Disease absent
            c = contingency.iloc[0, 1]  # Factor absent, Disease present
            d = contingency.iloc[0, 0]  # Factor absent, Disease absent

            odds_ratio = (a * d) / (b * c)

            # Calculate confidence interval
            log_or = np.log(odds_ratio)
            se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            ci_lower = np.exp(log_or - 1.96 * se_log_or)
            ci_upper = np.exp(log_or + 1.96 * se_log_or)

            # Fisher's exact test for p-value
            _, p_value = stats.fisher_exact(contingency)

            # Store results
            odds_ratios[factor] = odds_ratio
            confidence_intervals[factor] = (ci_lower, ci_upper)
            p_values[factor] = p_value

    # Sort risk factors by odds ratio
    sorted_factors = sorted(odds_ratios.keys(), key=lambda x: odds_ratios[x], reverse=True)

    # Create forest plot
    plt.figure(figsize=(12, 8))

    # Plot odds ratios and confidence intervals
    y_pos = np.arange(len(sorted_factors))

    for i, factor in enumerate(sorted_factors):
        # Plot odds ratio
        plt.plot(odds_ratios[factor], y_pos[i], 'o', markersize=10,
                 color='blue' if p_values[factor] < 0.05 else 'gray')

        # Plot confidence interval
        plt.plot([confidence_intervals[factor][0], confidence_intervals[factor][1]],
                 [y_pos[i], y_pos[i]], '-', linewidth=2,
                 color='blue' if p_values[factor] < 0.05 else 'gray')

        # Add odds ratio value
        plt.text(odds_ratios[factor] + 0.2, y_pos[i],
                 f"OR = {odds_ratios[factor]:.2f} ({confidence_intervals[factor][0]:.2f}-{confidence_intervals[factor][1]:.2f})",
                 va='center', ha='left', fontsize=10)

        # Add p-value
        if p_values[factor] < 0.001:
            p_text = "p < 0.001"
        else:
            p_text = f"p = {p_values[factor]:.3f}"

        plt.text(confidence_intervals[factor][1] + 1, y_pos[i],
                 p_text + (" *" if p_values[factor] < 0.05 else ""),
                 va='center', ha='left', fontsize=10,
                 color='blue' if p_values[factor] < 0.05 else 'gray')

    # Add vertical line at OR=1
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)

    # Set labels and title
    plt.yticks(y_pos, sorted_factors)
    plt.xlabel('Odds Ratio (log scale)', fontsize=14)
    plt.title('Risk Factors for Heart Disease (Forest Plot)', fontsize=16)

    # Use log scale for x-axis
    plt.xscale('log')

    # Set reasonable x limits
    plt.xlim(0.1, max([confidence_intervals[f][1] for f in sorted_factors]) * 2)

    # Add a legend for significance
    significance_patch = mpatches.Patch(color='blue', label='Statistically Significant (p < 0.05)')
    non_significance_patch = mpatches.Patch(color='gray', label='Not Statistically Significant')
    plt.legend(handles=[significance_patch, non_significance_patch], loc='lower right')

    # Add clinical interpretation
    plt.figtext(0.5, 0.01,
                "Clinical Interpretation: Several clinical factors are strongly associated with heart disease risk. "
                "Values greater than 1 indicate increased risk, with wider confidence intervals indicating less precise estimates.",
                ha='center', fontsize=11, style='italic',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Save the figure
    output_path = os.path.join(output_dir, 'risk_factors_forest_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Risk factor forest plot created and saved to {output_path}")
    return output_path


def document_visualization_choices(output_dir: str) -> str:
    """
    Document visualization choices and their analytical value.

    Parameters
    ----------
    output_dir : str
        Directory to save the documentation

    Returns
    -------
    str
        Path to the saved documentation
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create documentation content
    documentation = """# Medical Visualization Documentation

## Overview
This document explains the design choices and analytical value of the specialized medical visualizations created for the Heart Disease Dataset analysis.

## 1. Advanced Pairplot
**Design Choices:**
- Enhanced correlation visualization with statistics for both overall data and disease subgroups
- Corner plot to focus on lower triangle and increase readability
- Added statistical annotations to convey relationships quantitatively
- Medical-specific color palette to distinguish disease status

**Analytical Value:**
- Reveals complex relationships between key clinical parameters
- Allows simultaneous assessment of multiple clinical indicators
- Highlights differential correlations within disease vs. non-disease populations
- Supports identification of potential biomarkers for further investigation

## 2. Faceted Plots
**Design Choices:**
- Stratification by multiple clinically relevant factors (age group, gender, disease status)
- Enhanced statistical annotations including sample sizes and significance tests
- Maintained consistent scales within facet groups for valid comparisons
- Clinical interpretations provided to contextualize findings

**Analytical Value:**
- Enables identification of complex interaction effects between demographic and clinical variables
- Reveals subgroup-specific patterns that might be obscured in aggregate data
- Supports personalized risk assessment based on multiple factors
- Facilitates comparison of disease manifestation across different patient populations

## 3. Enhanced Violin Plots
**Design Choices:**
- Combined violins with individual data points to show both distribution and raw data
- Added statistical tests with effect sizes to quantify group differences
- Split violins to compare disease status within categorical variables
- Supplemented with density plots to better visualize distribution shapes

**Analytical Value:**
- Reveals the full distribution shape beyond simple measures of central tendency
- Shows outliers while maintaining context of the overall distribution
- Allows assessment of not just mean differences but variability differences between groups
- Provides clinically interpretable effect sizes to gauge practical significance

## 4. Clinical Risk Forest Plot
**Design Choices:**
- Forest plot format familiar to medical professionals
- Logarithmic scale to appropriately display odds ratios
- Confidence intervals to represent precision of estimates
- Color-coding to indicate statistical significance
- Sorted by effect size to emphasize strongest risk factors

**Analytical Value:**
- Quantifies the strength of association between binary risk factors and disease outcome
- Shows uncertainty in risk estimates through confidence intervals
- Allows for direct comparison of multiple risk factors in a single visualization
- Supports clinical decision-making by identifying strongest predictors of disease

## Standardization Principles
- Consistent color scheme across all visualizations
- Statistical annotations on all plots
- Clinical interpretations provided for all visualization types
- Typography chosen for readability in clinical settings (presentations, papers)
- Publication-quality resolution suitable for medical journals

## Limitations and Considerations
- Sample sizes in some subgroups may limit reliability of statistical tests
- Observational nature of the dataset limits causal inference
- Retrospective bias may influence the strength of some associations
- Clinical interpretations should be validated by medical professionals

"""

    # Save documentation to file
    output_path = os.path.join(output_dir, 'medical_visualization_documentation.md')
    with open(output_path, 'w') as f:
        f.write(documentation)

    logging.info(f"Visualization documentation created and saved to {output_path}")
    return output_path


def create_medical_visualizations(df: pd.DataFrame, output_dir: str) -> Dict:
    """
    Create all medical visualizations for the heart disease dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the variables
    output_dir : str
        Directory to save the visualizations

    Returns
    -------
    Dict
        Dictionary containing information about created visualizations
    """
    # Set up the visualization environment
    setup_viz_environment()

    # Dictionary to store visualization information
    viz_info = {}

    # Create pair plots
    logging.info("Creating advanced pairplot...")
    pairplot_path = create_advanced_pairplot(df, output_dir)
    viz_info['pairplot'] = pairplot_path

    # Create faceted plots
    logging.info("Creating faceted plots...")
    faceted_plots = create_faceted_plots(df, output_dir)
    viz_info['faceted_plots'] = faceted_plots

    # Create violin plots
    logging.info("Creating enhanced violin plots...")
    violin_plots = create_violin_plots(df, output_dir)
    viz_info['violin_plots'] = violin_plots

    # Create clinical risk visualization
    logging.info("Creating clinical risk visualization...")
    risk_plot = create_clinical_risk_visualization(df, output_dir)
    viz_info['risk_plot'] = risk_plot

    # Document visualization choices
    logging.info("Documenting visualization choices...")
    documentation_path = document_visualization_choices(output_dir)
    viz_info['documentation'] = documentation_path

    logging.info(f"Created all medical visualizations and saved to {output_dir}")
    return viz_info


if __name__ == "__main__":
    # Test the module if run directly
    print("Medical Visualization Module for the Heart Disease dataset.")
    print("This module provides specialized medical visualization functions.")
    print("Import and use in another script or notebook.")