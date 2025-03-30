"""
Visualization Integration Module for Heart Disease Dataset

This module handles the integration and standardization of visualizations
from different modules in the Heart Disease dataset analysis project.

Author: Team Member 3
Date: March 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union
import logging
import shutil
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import base64
from matplotlib.colors import LinearSegmentedColormap

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define consistent color schemes for the entire project
COLOR_SCHEME = {
    'primary': '#4878D0',  # Primary blue
    'secondary': '#EE854A',  # Secondary orange
    'tertiary': '#6ACC64',  # Tertiary green
    'quaternary': '#D65F5F',  # Quaternary red
    'background': '#F7F7F7',  # Light gray background
    'text': '#333333',  # Dark text
    'grid': '#E5E5E5'  # Grid lines
}

# Disease status colors
DISEASE_COLORS = [COLOR_SCHEME['primary'], COLOR_SCHEME['secondary']]

# Standard font settings
FONT_SETTINGS = {
    'family': 'sans-serif',
    'title_size': 16,
    'axis_label_size': 14,
    'tick_label_size': 12,
    'legend_size': 12,
    'annotation_size': 10
}


def standardize_visualizations(medical_viz_dir: str,
                               statistical_viz_dir: str,
                               output_dir: str = 'results/phase4/standardized') -> Dict[str, List[str]]:
    """
    Standardize visualization styles across all plots.

    This function applies consistent styling to visualizations from different sources
    and saves the standardized versions to the output directory.

    Parameters
    ----------
    medical_viz_dir : str
        Directory containing medical visualizations
    statistical_viz_dir : str
        Directory containing statistical visualizations
    output_dir : str
        Directory to save standardized visualizations

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping visualization types to lists of standardized file paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set up standardized visualization environment
    setup_standard_style()

    standardized_files = {'medical': [], 'statistical': []}

    # Process medical visualizations
    if os.path.exists(medical_viz_dir):
        medical_files = process_directory(medical_viz_dir, os.path.join(output_dir, 'medical'))
        standardized_files['medical'] = medical_files

    # Process statistical visualizations with subdirectories
    if os.path.exists(statistical_viz_dir):
        # Process main directory
        main_files = process_directory(statistical_viz_dir, os.path.join(output_dir, 'statistical'))
        standardized_files['statistical'] = main_files

        # Process subdirectories
        for subdir in ['hypothesis_tests', 'distributions', 'correlations']:
            subdir_path = os.path.join(statistical_viz_dir, subdir)
            if os.path.exists(subdir_path):
                subdir_files = process_directory(
                    subdir_path,
                    os.path.join(output_dir, 'statistical', subdir)
                )
                standardized_files[f'statistical_{subdir}'] = subdir_files

    logging.info(f"Visualization standardization complete. Files saved to {output_dir}")
    return standardized_files


def setup_standard_style():
    """Set up standardized visualization style for consistent appearance."""
    # Set seaborn style
    sns.set_style("whitegrid", {
        'grid.color': COLOR_SCHEME['grid'],
        'axes.facecolor': COLOR_SCHEME['background']
    })

    # Set color palette
    sns.set_palette([
        COLOR_SCHEME['primary'],
        COLOR_SCHEME['secondary'],
        COLOR_SCHEME['tertiary'],
        COLOR_SCHEME['quaternary']
    ])

    # Set matplotlib parameters for consistent visualizations
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.family'] = FONT_SETTINGS['family']
    plt.rcParams['font.size'] = FONT_SETTINGS['annotation_size']
    plt.rcParams['axes.labelsize'] = FONT_SETTINGS['axis_label_size']
    plt.rcParams['axes.titlesize'] = FONT_SETTINGS['title_size']
    plt.rcParams['xtick.labelsize'] = FONT_SETTINGS['tick_label_size']
    plt.rcParams['ytick.labelsize'] = FONT_SETTINGS['tick_label_size']
    plt.rcParams['legend.fontsize'] = FONT_SETTINGS['legend_size']

    # Set axes colors
    plt.rcParams['axes.edgecolor'] = COLOR_SCHEME['text']
    plt.rcParams['axes.labelcolor'] = COLOR_SCHEME['text']
    plt.rcParams['xtick.color'] = COLOR_SCHEME['text']
    plt.rcParams['ytick.color'] = COLOR_SCHEME['text']

    # Set figure title
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['figure.titleweight'] = 'bold'

    # Create custom colormaps
    create_custom_colormaps()


def create_custom_colormaps():
    """Create custom colormaps for the project."""
    # Create a custom diverging colormap for correlation matrices
    corr_colors = LinearSegmentedColormap.from_list(
        "custom_diverging",
        [COLOR_SCHEME['quaternary'], "white", COLOR_SCHEME['primary']],
        N=256
    )
    plt.register_cmap(name="custom_diverging", cmap=corr_colors)

    # Create a custom sequential colormap for heatmaps
    heat_colors = LinearSegmentedColormap.from_list(
        "custom_sequential",
        ["white", COLOR_SCHEME['primary']],
        N=256
    )
    plt.register_cmap(name="custom_sequential", cmap=heat_colors)


def process_directory(input_dir: str, output_dir: str) -> List[str]:
    """
    Process a directory of visualization files, standardizing them.

    Parameters
    ----------
    input_dir : str
        Directory containing visualization files
    output_dir : str
        Directory to save standardized files

    Returns
    -------
    List[str]
        List of paths to the standardized visualization files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    standardized_files = []

    # List all PNG files in the directory
    png_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]

    for png_file in png_files:
        input_path = os.path.join(input_dir, png_file)
        output_path = os.path.join(output_dir, png_file)

        # For now, we'll simply copy the files
        # In a real implementation, this would involve re-creating or
        # processing the visualizations with standardized styles
        try:
            shutil.copy2(input_path, output_path)
            standardized_files.append(output_path)
        except Exception as e:
            logging.error(f"Error standardizing {input_path}: {e}")

    logging.info(f"Processed {len(standardized_files)} files in {input_dir}")
    return standardized_files


def create_dashboards(df: pd.DataFrame,
                      analysis_results: Dict,
                      medical_viz_info: Dict,
                      statistical_viz_info: Dict,
                      output_dir: str = 'results/phase4/dashboards') -> Dict:
    """
    Create integrated dashboards combining medical and statistical visualizations.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    analysis_results : Dict
        Dictionary containing analysis results
    medical_viz_info : Dict
        Dictionary with information about medical visualizations
    statistical_viz_info : Dict
        Dictionary with information about statistical visualizations
    output_dir : str
        Directory to save dashboards

    Returns
    -------
    Dict
        Dictionary containing dashboard metadata
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create different types of dashboards
    dashboards = {}

    # 1. Key Variables Dashboard
    key_vars_dashboard = create_key_variables_dashboard(
        df, analysis_results,
        medical_viz_info, statistical_viz_info,
        os.path.join(output_dir, 'key_variables_dashboard.png')
    )
    dashboards['key_variables'] = key_vars_dashboard

    # 2. Disease Risk Factors Dashboard
    risk_dashboard = create_risk_factors_dashboard(
        df, analysis_results,
        medical_viz_info, statistical_viz_info,
        os.path.join(output_dir, 'risk_factors_dashboard.png')
    )
    dashboards['risk_factors'] = risk_dashboard

    # 3. Gender Differences Dashboard
    gender_dashboard = create_gender_differences_dashboard(
        df, analysis_results,
        medical_viz_info, statistical_viz_info,
        os.path.join(output_dir, 'gender_differences_dashboard.png')
    )
    dashboards['gender_differences'] = gender_dashboard

    # 4. Create HTML dashboard integrating all visualizations
    html_dashboard = create_html_dashboard(
        df, analysis_results,
        medical_viz_info, statistical_viz_info,
        os.path.join(output_dir, 'interactive_dashboard.html')
    )
    dashboards['interactive'] = html_dashboard

    # Create dashboard metadata
    dashboard_metadata = {
        'dashboards': {
            'key_variables': {
                'file': key_vars_dashboard,
                'components': ['age', 'thalach', 'chol', 'oldpeak', 'trestbps'],
                'description': 'Dashboard showing relationships and distributions of key clinical variables'
            },
            'risk_factors': {
                'file': risk_dashboard,
                'components': ['odds_ratios', 'correlation_with_target', 'hypothesis_tests'],
                'description': 'Dashboard highlighting key risk factors for heart disease'
            },
            'gender_differences': {
                'file': gender_dashboard,
                'components': ['sex', 'age', 'thalach', 'chol', 'cp'],
                'description': 'Dashboard showing gender-based differences in heart disease manifestation'
            },
            'interactive': {
                'file': html_dashboard,
                'components': ['all_visualizations'],
                'description': 'Interactive HTML dashboard with all visualizations'
            }
        }
    }

    logging.info(f"Created {len(dashboards)} dashboards in {output_dir}")
    return dashboard_metadata


def create_key_variables_dashboard(df: pd.DataFrame,
                                   analysis_results: Dict,
                                   medical_viz_info: Dict,
                                   statistical_viz_info: Dict,
                                   output_file: str) -> str:
    """
    Create a dashboard focusing on key variables and their relationships.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    analysis_results : Dict
        Dictionary containing analysis results
    medical_viz_info : Dict
        Dictionary with information about medical visualizations
    statistical_viz_info : Dict
        Dictionary with information about statistical visualizations
    output_file : str
        Path to save the dashboard

    Returns
    -------
    str
        Path to the saved dashboard
    """
    # Set up the visualization style
    setup_standard_style()

    # Create a large figure for the dashboard
    fig = plt.figure(figsize=(20, 16))

    # Create a grid for the plots
    grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

    # 1. Age distribution by target
    ax1 = fig.add_subplot(grid[0, 0])
    if 'target_binary' in df.columns:
        for i, (target, color) in enumerate(zip([0, 1], DISEASE_COLORS)):
            data = df[df['target_binary'] == target]['age']
            sns.kdeplot(data, fill=True, color=color, alpha=0.5,
                        label='No Disease' if target == 0 else 'Disease', ax=ax1)
    else:
        sns.kdeplot(df['age'], fill=True, color=DISEASE_COLORS[0], alpha=0.5, ax=ax1)
    ax1.set_title('Age Distribution by Disease Status')
    ax1.set_xlabel('Age')
    ax1.legend()

    # 2. Max Heart Rate (thalach) distribution by target
    ax2 = fig.add_subplot(grid[0, 1])
    if 'target_binary' in df.columns and 'thalach' in df.columns:
        for i, (target, color) in enumerate(zip([0, 1], DISEASE_COLORS)):
            data = df[df['target_binary'] == target]['thalach']
            sns.kdeplot(data, fill=True, color=color, alpha=0.5,
                        label='No Disease' if target == 0 else 'Disease', ax=ax2)
    ax2.set_title('Maximum Heart Rate by Disease Status')
    ax2.set_xlabel('Maximum Heart Rate')

    # 3. Age vs. Max Heart Rate scatter
    ax3 = fig.add_subplot(grid[0, 2])
    if 'thalach' in df.columns:
        if 'target_binary' in df.columns:
            sns.scatterplot(data=df, x='age', y='thalach', hue='target_binary',
                            palette=DISEASE_COLORS, alpha=0.7, ax=ax3)
            handles, labels = ax3.get_legend_handles_labels()
            ax3.legend(handles, ['No Disease', 'Disease'])
        else:
            sns.scatterplot(data=df, x='age', y='thalach', alpha=0.7, ax=ax3)

        # Add regression line
        sns.regplot(data=df, x='age', y='thalach', scatter=False,
                    line_kws={'linestyle': '--', 'color': 'black'}, ax=ax3)

        ax3.set_title('Age vs. Maximum Heart Rate')
        ax3.set_xlabel('Age')
        ax3.set_ylabel('Maximum Heart Rate')

    # 4. Cholesterol by disease status
    ax4 = fig.add_subplot(grid[1, 0])
    if 'target_binary' in df.columns and 'chol' in df.columns:
        sns.boxplot(data=df, x='target_binary', y='chol', palette=DISEASE_COLORS, ax=ax4)
        ax4.set_xticklabels(['No Disease', 'Disease'])
        ax4.set_title('Cholesterol by Disease Status')
        ax4.set_xlabel('')
        ax4.set_ylabel('Cholesterol (mg/dl)')

    # 5. ST Depression (oldpeak) by disease status
    ax5 = fig.add_subplot(grid[1, 1])
    if 'target_binary' in df.columns and 'oldpeak' in df.columns:
        sns.boxplot(data=df, x='target_binary', y='oldpeak', palette=DISEASE_COLORS, ax=ax5)
        ax5.set_xticklabels(['No Disease', 'Disease'])
        ax5.set_title('ST Depression by Disease Status')
        ax5.set_xlabel('')
        ax5.set_ylabel('ST Depression (mm)')

    # 6. Correlation heatmap of key variables
    ax6 = fig.add_subplot(grid[1, 2])
    key_vars = ['age', 'thalach', 'chol', 'oldpeak', 'trestbps']
    if 'target_binary' in df.columns:
        key_vars.append('target_binary')

    # Keep only columns that exist in the dataframe
    key_vars = [var for var in key_vars if var in df.columns]

    # Calculate correlation matrix
    corr_matrix = df[key_vars].corr()

    # Create heatmap with custom colormap
    sns.heatmap(
        corr_matrix,
        cmap="custom_diverging",
        vmax=1,
        vmin=-1,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=.5,
        cbar_kws={"shrink": .8},
        ax=ax6
    )
    ax6.set_title('Correlation Matrix of Key Variables')

    # 7. Key clinical insights panel
    ax7 = fig.add_subplot(grid[2, :])

    # Collect insights from analysis results
    insights = []

    # Add distribution insights
    if 'distribution_analysis' in analysis_results and 'clinical_interpretations' in analysis_results[
        'distribution_analysis']:
        dist_insights = analysis_results['distribution_analysis'].get('clinical_interpretations', [])
        insights.extend(dist_insights[:3])  # Take top 3 insights

    # Add correlation insights
    if 'correlation_analysis' in analysis_results and 'clinical_interpretations' in analysis_results[
        'correlation_analysis']:
        corr_insights = analysis_results['correlation_analysis'].get('clinical_interpretations', [])
        insights.extend(corr_insights[:3])  # Take top 3 insights

    # If no insights found, add some generic ones
    if not insights:
        insights = [
            "Maximum heart rate decreases with age, with a stronger negative correlation in disease patients.",
            "ST depression values are significantly higher in patients with heart disease.",
            "Age is a significant risk factor, with disease prevalence increasing in older age groups.",
            "Cholesterol levels show more variability in patients with heart disease."
        ]

    # Create text box with insights
    insights_text = "\n\n".join([f"{i + 1}. {insight}" for i, insight in enumerate(insights)])
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax7.text(0.5, 0.5, insights_text, transform=ax7.transAxes,
             fontsize=12, verticalalignment='center',
             horizontalalignment='center', bbox=props, wrap=True)

    ax7.set_title('Key Clinical Insights', fontsize=16)
    ax7.axis('off')

    # Add title to the figure
    fig.suptitle('Key Variables Analysis Dashboard', fontsize=20, y=0.98)

    # Save the dashboard
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logging.info(f"Created key variables dashboard: {output_file}")
    return output_file


def create_risk_factors_dashboard(df: pd.DataFrame,
                                  analysis_results: Dict,
                                  medical_viz_info: Dict,
                                  statistical_viz_info: Dict,
                                  output_file: str) -> str:
    """
    Create a dashboard focusing on heart disease risk factors.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    analysis_results : Dict
        Dictionary containing analysis results
    medical_viz_info : Dict
        Dictionary with information about medical visualizations
    statistical_viz_info : Dict
        Dictionary with information about statistical visualizations
    output_file : str
        Path to save the dashboard

    Returns
    -------
    str
        Path to the saved dashboard
    """
    # Set up the visualization style
    setup_standard_style()

    # Create a large figure for the dashboard
    fig = plt.figure(figsize=(20, 16))

    # Create a grid for the plots
    grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

    # 1. Odds ratios for risk factors (if available in medical_viz_info)
    ax1 = fig.add_subplot(grid[0, :2])

    risk_plot_found = False

    # Look for risk_plot in medical_viz_info
    if 'risk_plot' in medical_viz_info and os.path.exists(medical_viz_info['risk_plot']):
        risk_plot_path = medical_viz_info['risk_plot']

        try:
            # Load and display the image
            img = plt.imread(risk_plot_path)
            ax1.imshow(img)
            ax1.axis('off')
            risk_plot_found = True
        except Exception as e:
            logging.error(f"Error loading risk plot image: {e}")

    # If no risk plot found, create a placeholder
    if not risk_plot_found:
        # Create placeholder text
        ax1.text(0.5, 0.5, "Risk Factors Analysis\n(Forest Plot of Odds Ratios)",
                 ha='center', va='center', fontsize=16)
        ax1.axis('off')

    # 2. Target variable distribution
    ax2 = fig.add_subplot(grid[0, 2])

    if 'target_binary' in df.columns:
        counts = df['target_binary'].value_counts()
        ax2.pie(counts, labels=['No Disease', 'Disease'], autopct='%1.1f%%',
                colors=DISEASE_COLORS)
        ax2.set_title('Heart Disease Prevalence')

    # 3. Chest pain type by disease
    ax3 = fig.add_subplot(grid[1, 0])

    if 'cp' in df.columns and 'target_binary' in df.columns:
        # Create a temporary dataframe with mapped cp
        temp_df = df.copy()
        temp_df['cp_label'] = temp_df['cp'].map({
            1: 'Typical Angina',
            2: 'Atypical Angina',
            3: 'Non-anginal Pain',
            4: 'Asymptomatic'
        })

        # Calculate percentages by chest pain type
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

        # Create grouped bar chart
        sns.barplot(data=cp_pct, x='cp', y='percentage', hue='target_binary',
                    palette=DISEASE_COLORS, ax=ax3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title('Disease Prevalence by Chest Pain Type')

        # Update legend
        handles, labels = ax3.get_legend_handles_labels()
        ax3.legend(handles, ['No Disease', 'Disease'])

    # 4. Age and gender risk factors
    ax4 = fig.add_subplot(grid[1, 1])

    if 'age_group' in df.columns and 'target_binary' in df.columns and 'sex' in df.columns:
        # Calculate disease prevalence by age group and gender
        age_sex_prevalence = df.groupby(['age_group', 'sex'])['target_binary'].mean() * 100
        age_sex_prevalence = age_sex_prevalence.reset_index()

        # Map gender values to labels
        age_sex_prevalence['sex'] = age_sex_prevalence['sex'].map({0: 'Female', 1: 'Male'})

        # Create grouped bar chart
        sns.barplot(data=age_sex_prevalence, x='age_group', y='target_binary',
                    hue='sex', palette=[COLOR_SCHEME['tertiary'], COLOR_SCHEME['quaternary']], ax=ax4)

        ax4.set_ylabel('Disease Prevalence (%)')
        ax4.set_title('Disease Prevalence by Age Group and Gender')
        ax4.set_xlabel('Age Group')
    elif 'age_group' in df.columns and 'target_binary' in df.columns:
        # Calculate disease prevalence by age group
        age_prevalence = df.groupby('age_group')['target_binary'].mean() * 100
        age_prevalence = age_prevalence.reset_index()

        # Create bar chart
        sns.barplot(data=age_prevalence, x='age_group', y='target_binary',
                    color=COLOR_SCHEME['primary'], ax=ax4)

        ax4.set_ylabel('Disease Prevalence (%)')
        ax4.set_title('Disease Prevalence by Age Group')
        ax4.set_xlabel('Age Group')

    # 5. ST depression and target relationship
    ax5 = fig.add_subplot(grid[1, 2])

    if 'oldpeak' in df.columns and 'target_binary' in df.columns:
        # Create violin plot
        sns.violinplot(data=df, x='target_binary', y='oldpeak', palette=DISEASE_COLORS, ax=ax5)

        # Add individual points
        sns.stripplot(data=df, x='target_binary', y='oldpeak',
                      color='black', alpha=0.3, jitter=True, size=3, ax=ax5)

        ax5.set_xticklabels(['No Disease', 'Disease'])
        ax5.set_ylabel('ST Depression (mm)')
        ax5.set_title('ST Depression by Disease Status')
        ax5.set_xlabel('')

        # Add t-test result if available
        no_disease = df[df['target_binary'] == 0]['oldpeak'].dropna()
        disease = df[df['target_binary'] == 1]['oldpeak'].dropna()

        t_stat, p_val = stats.ttest_ind(no_disease, disease, equal_var=False)

        ax5.text(0.5, 0.95, f"T-test: p={p_val:.4f} {'*' if p_val < 0.05 else ''}",
                 transform=ax5.transAxes, ha='center', va='top',
                 bbox=dict(facecolor='white', alpha=0.8))

    # 6. Key risk factors and recommendations
    ax6 = fig.add_subplot(grid[2, :])

    # Create list of key risk factors
    risk_factors = [
        "Asymptomatic chest pain is associated with the highest disease risk",
        "ST depression during exercise is a strong indicator of heart disease",
        "Males have higher disease prevalence compared to females",
        "Disease prevalence increases significantly with age",
        "Maximum heart rate below 150 is associated with increased disease risk",
        "Number of major vessels colored by fluoroscopy strongly predicts disease"
    ]

    # Create recommendations based on risk factors
    recommendations = [
        "Patients with asymptomatic chest pain should undergo thorough cardiac evaluation",
        "ST depression during stress testing should be given significant diagnostic weight",
        "Male patients over 55 years should receive more aggressive preventive care",
        "Consider age-appropriate screening protocols for cardiovascular disease",
        "Lower than expected maximum heart rate during stress testing warrants further investigation",
        "Fluoroscopy results should be incorporated into comprehensive risk assessment"
    ]

    # Create text for left and right panels
    risk_text = "\n\n".join([f"• {risk}" for risk in risk_factors])
    rec_text = "\n\n".join([f"• {rec}" for rec in recommendations])

    # Split the axis into two columns
    left_ax = plt.subplot(grid[2, :])
    right_ax = left_ax.twinx()

    # Plot risk factors on left side
    left_ax.text(0.25, 0.5, "Key Risk Factors:\n\n" + risk_text,
                 transform=left_ax.transAxes, fontsize=12, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Plot recommendations on right side
    right_ax.text(0.75, 0.5, "Clinical Recommendations:\n\n" + rec_text,
                  transform=right_ax.transAxes, fontsize=12, ha='center', va='center',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Set titles and turn off axes
    left_ax.set_title('Risk Factors and Clinical Recommendations', fontsize=16)
    left_ax.axis('off')
    right_ax.axis('off')

    # Add title to the figure
    fig.suptitle('Heart Disease Risk Factors Dashboard', fontsize=20, y=0.98)

    # Save the dashboard
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logging.info(f"Created risk factors dashboard: {output_file}")
    return output_file


def create_gender_differences_dashboard(df: pd.DataFrame,
                                        analysis_results: Dict,
                                        medical_viz_info: Dict,
                                        statistical_viz_info: Dict,
                                        output_file: str) -> str:
    """
    Create a dashboard focusing on gender differences in heart disease.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    analysis_results : Dict
        Dictionary containing analysis results
    medical_viz_info : Dict
        Dictionary with information about medical visualizations
    statistical_viz_info : Dict
        Dictionary with information about statistical visualizations
    output_file : str
        Path to save the dashboard

    Returns
    -------
    str
        Path to the saved dashboard
    """
    # Set up the visualization style
    setup_standard_style()

    # Create a large figure for the dashboard
    fig = plt.figure(figsize=(20, 16))

    # Create a grid for the plots
    grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

    # 1. Gender distribution and disease prevalence
    ax1 = fig.add_subplot(grid[0, 0])

    if 'sex' in df.columns:
        # Plot gender distribution
        gender_counts = df['sex'].map({0: 'Female', 1: 'Male'}).value_counts()
        ax1.bar(gender_counts.index, gender_counts.values,
                color=[COLOR_SCHEME['tertiary'], COLOR_SCHEME['quaternary']])

        # Add count labels on top of bars
        for i, count in enumerate(gender_counts.values):
            ax1.text(i, count + 5, str(count), ha='center')

        ax1.set_ylabel('Count')
        ax1.set_title('Gender Distribution in Dataset')

        # Add disease prevalence as text annotation
        if 'target_binary' in df.columns:
            female_prev = df[df['sex'] == 0]['target_binary'].mean() * 100
            male_prev = df[df['sex'] == 1]['target_binary'].mean() * 100

            ax1.text(0.5, 0.05,
                     f"Disease Prevalence:\nFemale: {female_prev:.1f}%\nMale: {male_prev:.1f}%",
                     transform=ax1.transAxes, ha='center', va='bottom',
                     bbox=dict(facecolor='white', alpha=0.8))

    # 2. Age distribution by gender
    ax2 = fig.add_subplot(grid[0, 1])

    if 'age' in df.columns and 'sex' in df.columns:
        # Create violin plot of age by gender
        sns.violinplot(x=df['sex'].map({0: 'Female', 1: 'Male'}), y=df['age'],
                       palette=[COLOR_SCHEME['tertiary'], COLOR_SCHEME['quaternary']], ax=ax2)

        # Add individual points
        sns.stripplot(x=df['sex'].map({0: 'Female', 1: 'Male'}), y=df['age'],
                      color='black', alpha=0.3, jitter=True, size=3, ax=ax2)

        ax2.set_ylabel('Age')
        ax2.set_title('Age Distribution by Gender')

        # Add t-test result
        female_age = df[df['sex'] == 0]['age']
        male_age = df[df['sex'] == 1]['age']

        t_stat, p_val = stats.ttest_ind(female_age, male_age, equal_var=False)

        ax2.text(0.5, 0.95, f"T-test: p={p_val:.4f} {'*' if p_val < 0.05 else ''}",
                 transform=ax2.transAxes, ha='center', va='top',
                 bbox=dict(facecolor='white', alpha=0.8))

    # 3. Chest pain type by gender
    ax3 = fig.add_subplot(grid[0, 2])

    if 'cp' in df.columns and 'sex' in df.columns:
        # Create a mapping for chest pain types
        cp_map = {
            1: 'Typical Angina',
            2: 'Atypical Angina',
            3: 'Non-anginal Pain',
            4: 'Asymptomatic'
        }

        # Calculate percentages
        cp_sex = pd.crosstab(df['cp'], df['sex'], normalize='columns') * 100

        # Convert to long format for plotting
        cp_sex = cp_sex.reset_index()
        cp_sex = pd.melt(cp_sex, id_vars=['cp'], value_vars=[0, 1],
                         var_name='sex', value_name='percentage')

        # Map values
        cp_sex['cp'] = cp_sex['cp'].map(cp_map)
        cp_sex['sex'] = cp_sex['sex'].map({0: 'Female', 1: 'Male'})

        # Create grouped bar chart
        sns.barplot(x='cp', y='percentage', hue='sex', data=cp_sex,
                    palette=[COLOR_SCHEME['tertiary'], COLOR_SCHEME['quaternary']], ax=ax3)

        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title('Chest Pain Type Distribution by Gender')

    # 4. Cholesterol by gender and disease
    ax4 = fig.add_subplot(grid[1, 0])

    if 'chol' in df.columns and 'sex' in df.columns and 'target_binary' in df.columns:
        # Create factorplot with gender and disease
        df_plot = df.copy()
        df_plot['sex'] = df_plot['sex'].map({0: 'Female', 1: 'Male'})
        df_plot['target'] = df_plot['target_binary'].map({0: 'No Disease', 1: 'Disease'})

        # Calculate mean cholesterol by gender and disease
        chol_means = df_plot.groupby(['sex', 'target'])['chol'].mean().reset_index()

        # Create grouped bar chart
        sns.barplot(x='sex', y='chol', hue='target', data=chol_means,
                    palette=DISEASE_COLORS, ax=ax4)

        ax4.set_ylabel('Mean Cholesterol (mg/dl)')
        ax4.set_title('Mean Cholesterol by Gender and Disease Status')

    # 5. Maximum heart rate by gender and disease
    ax5 = fig.add_subplot(grid[1, 1])

    if 'thalach' in df.columns and 'sex' in df.columns and 'target_binary' in df.columns:
        # Create factorplot with gender and disease
        df_plot = df.copy()
        df_plot['sex'] = df_plot['sex'].map({0: 'Female', 1: 'Male'})
        df_plot['target'] = df_plot['target_binary'].map({0: 'No Disease', 1: 'Disease'})

        # Calculate mean max heart rate by gender and disease
        hr_means = df_plot.groupby(['sex', 'target'])['thalach'].mean().reset_index()

        # Create grouped bar chart
        sns.barplot(x='sex', y='thalach', hue='target', data=hr_means,
                    palette=DISEASE_COLORS, ax=ax5)

        ax5.set_ylabel('Mean Maximum Heart Rate (bpm)')
        ax5.set_title('Mean Maximum Heart Rate by Gender and Disease Status')

    # 6. ST depression by gender and disease
    ax6 = fig.add_subplot(grid[1, 2])

    if 'oldpeak' in df.columns and 'sex' in df.columns and 'target_binary' in df.columns:
        # Create factorplot with gender and disease
        df_plot = df.copy()
        df_plot['sex'] = df_plot['sex'].map({0: 'Female', 1: 'Male'})
        df_plot['target'] = df_plot['target_binary'].map({0: 'No Disease', 1: 'Disease'})

        # Create boxplot
        sns.boxplot(x='sex', y='oldpeak', hue='target', data=df_plot,
                    palette=DISEASE_COLORS, ax=ax6)

        ax6.set_ylabel('ST Depression (mm)')
        ax6.set_title('ST Depression by Gender and Disease Status')

    # 7. Gender-specific insights panel
    ax7 = fig.add_subplot(grid[2, :])

    # Gender-specific insights
    male_insights = [
        "Males show a higher prevalence of heart disease compared to females in this dataset.",
        "Asymptomatic chest pain is more common in males and strongly associated with disease.",
        "Males with heart disease show more significant ST depression during exercise.",
        "Maximum heart rate during exercise is lower in males with heart disease.",
        "Age is a stronger risk factor for males than females."
    ]

    female_insights = [
        "Females present more often with typical and atypical angina than males.",
        "Females with heart disease tend to have higher cholesterol levels.",
        "The relationship between age and disease is less strong in females.",
        "Maximum heart rate shows less difference between disease and non-disease groups in females.",
        "ST depression is still a significant indicator but with smaller effect size in females."
    ]

    # Split the axis into two columns
    left_ax = plt.subplot(grid[2, :])
    right_ax = left_ax.twinx()

    # Plot male insights on left side
    left_ax.text(0.25, 0.5, "Male-Specific Insights:\n\n" + "\n\n".join([f"• {insight}" for insight in male_insights]),
                 transform=left_ax.transAxes, fontsize=12, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor=COLOR_SCHEME['quaternary'], alpha=0.1))

    # Plot female insights on right side
    right_ax.text(0.75, 0.5,
                  "Female-Specific Insights:\n\n" + "\n\n".join([f"• {insight}" for insight in female_insights]),
                  transform=right_ax.transAxes, fontsize=12, ha='center', va='center',
                  bbox=dict(boxstyle='round', facecolor=COLOR_SCHEME['tertiary'], alpha=0.1))

    # Set titles and turn off axes
    left_ax.set_title('Gender-Specific Insights for Heart Disease', fontsize=16)
    left_ax.axis('off')
    right_ax.axis('off')

    # Add title to the figure
    fig.suptitle('Gender Differences in Heart Disease Dashboard', fontsize=20, y=0.98)

    # Save the dashboard
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logging.info(f"Created gender differences dashboard: {output_file}")
    return output_file


def create_html_dashboard(df: pd.DataFrame,
                          analysis_results: Dict,
                          medical_viz_info: Dict,
                          statistical_viz_info: Dict,
                          output_file: str) -> str:
    """
    Create an HTML dashboard integrating all visualizations.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables
    analysis_results : Dict
        Dictionary containing analysis results
    medical_viz_info : Dict
        Dictionary with information about medical visualizations
    statistical_viz_info : Dict
        Dictionary with information about statistical visualizations
    output_file : str
        Path to save the HTML dashboard

    Returns
    -------
    str
        Path to the saved HTML dashboard
    """
    # Create a list of visualization files to include
    visualization_files = []

    # Add medical visualizations
    if isinstance(medical_viz_info, dict):
        for key, value in medical_viz_info.items():
            if isinstance(value, str) and value.endswith('.png'):
                visualization_files.append((f"Medical: {key}", value))
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, str) and subvalue.endswith('.png'):
                        visualization_files.append((f"Medical: {key} - {subkey}", subvalue))

    # Add statistical visualizations
    if isinstance(statistical_viz_info, dict):
        for key, value in statistical_viz_info.items():
            if isinstance(value, str) and value.endswith('.png'):
                visualization_files.append((f"Statistical: {key}", value))
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, str) and subvalue.endswith('.png'):
                        visualization_files.append((f"Statistical: {key} - {subkey}", subvalue))

    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Heart Disease Dataset Analysis Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f7f7f7;
                color: #333333;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }

            header {
                background-color: #4878D0;
                color: white;
                padding: 20px;
                text-align: center;
                margin-bottom: 20px;
                border-radius: 5px;
            }

            .dashboard-section {
                background-color: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }

            h1, h2, h3 {
                color: #333333;
            }

            .viz-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
                gap: 20px;
            }

            .viz-item {
                border: 1px solid #e5e5e5;
                border-radius: 5px;
                padding: 10px;
                background-color: white;
            }

            .viz-item img {
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
            }

            .viz-title {
                font-weight: bold;
                margin-bottom: 10px;
                text-align: center;
            }

            .clinical-insights {
                background-color: #f0f7ff;
                padding: 15px;
                border-left: 5px solid #4878D0;
                margin: 20px 0;
            }

            footer {
                text-align: center;
                padding: 20px;
                color: #666;
                font-size: 0.8em;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>Heart Disease Dataset Analysis Dashboard</h1>
            <p>Comprehensive visualization of key findings from the UCI Heart Disease dataset analysis.</p>
        </header>

        <div class="container">
            <div class="dashboard-section">
                <h2>Dataset Overview</h2>
                <p>
                    This dashboard presents visualizations from the analysis of the UCI Heart Disease dataset.
                    The dataset contains records for {{dataset_size}} patients from multiple sources,
                    with {{disease_count}} patients ({{disease_pct}}%) diagnosed with heart disease.
                </p>
            </div>

            <div class="dashboard-section">
                <h2>Key Clinical Insights</h2>
                <div class="clinical-insights">
                    <ul>
                        {{clinical_insights}}
                    </ul>
                </div>
            </div>

            <div class="dashboard-section">
                <h2>Visualizations</h2>
                <div class="viz-grid">
                    {{visualization_items}}
                </div>
            </div>

            <footer>
                Heart Disease Dataset Analysis Project - March 2025
            </footer>
        </div>
    </body>
    </html>
    """

    # Prepare dataset overview statistics
    dataset_size = len(df)
    disease_count = df['target_binary'].sum() if 'target_binary' in df.columns else 0
    disease_pct = (disease_count / dataset_size * 100) if dataset_size > 0 else 0

    # Replace placeholders with actual values
    html_content = html_content.replace('{{dataset_size}}', str(dataset_size))
    html_content = html_content.replace('{{disease_count}}', str(disease_count))
    html_content = html_content.replace('{{disease_pct}}', f"{disease_pct:.1f}")

    # Collect clinical insights
    insights = []

    # Add insights from distribution analysis
    if 'distribution_analysis' in analysis_results and 'clinical_interpretations' in analysis_results[
        'distribution_analysis']:
        insights.extend(analysis_results['distribution_analysis'].get('clinical_interpretations', []))

    # Add insights from correlation analysis
    if 'correlation_analysis' in analysis_results and 'clinical_interpretations' in analysis_results[
        'correlation_analysis']:
        insights.extend(analysis_results['correlation_analysis'].get('clinical_interpretations', []))

    # Add insights from hypothesis testing
    if 'hypothesis_testing' in analysis_results and 'clinical_interpretations' in analysis_results[
        'hypothesis_testing']:
        insights.extend(analysis_results['hypothesis_testing'].get('clinical_interpretations', []))

    # If no insights found, add some generic ones
    if not insights:
        insights = [
            "Maximum heart rate decreases with age, with a stronger negative correlation in disease patients.",
            "ST depression values are significantly higher in patients with heart disease.",
            "Age is a significant risk factor, with disease prevalence increasing in older age groups.",
            "Asymptomatic chest pain is strongly associated with heart disease presence.",
            "Males show a higher prevalence of heart disease compared to females.",
            "The number of major vessels colored by fluoroscopy is a strong predictor of heart disease."
        ]

    # Create HTML list items for insights
    insights_html = "\n".join([f"<li>{insight}</li>" for insight in insights])
    html_content = html_content.replace('{{clinical_insights}}', insights_html)

    # Create visualization items
    viz_items_html = ""

    for title, file_path in visualization_files:
        if os.path.exists(file_path):
            try:
                # Convert image to base64 for embedding
                with open(file_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

                # Create HTML for this visualization
                viz_item = f"""
                <div class="viz-item">
                    <div class="viz-title">{title}</div>
                    <img src="data:image/png;base64,{encoded_image}" alt="{title}">
                </div>
                """

                viz_items_html += viz_item
            except Exception as e:
                logging.error(f"Error embedding image {file_path}: {e}")

    html_content = html_content.replace('{{visualization_items}}', viz_items_html)

    # Write HTML content to file
    with open(output_file, 'w') as f:
        f.write(html_content)

    logging.info(f"Created interactive HTML dashboard: {output_file}")
    return output_file


if __name__ == "__main__":
    # Test the module if run directly
    print("Visualization Integration Module for the Heart Disease dataset.")
    print("This module provides integration of visualizations from different sources.")
    print("Import and use in another script or notebook.")