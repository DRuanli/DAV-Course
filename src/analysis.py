"""
Analysis Module for Heart Disease Dataset Analysis

This module handles statistical analysis functions and data exploration
for the UCI Heart Disease dataset.

Author: Team Member 2
Date: March 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import scipy.stats as stats


def get_variable_descriptions() -> Dict[str, str]:
    """
    Get descriptions for all variables in the heart disease dataset.

    Returns
    -------
    Dict[str, str]
        Dictionary of variable names and their descriptions
    """
    descriptions = {
        'age': 'Age of the patient in years',
        'sex': 'Gender of the patient (1 = male, 0 = female)',
        'cp': 'Chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic)',
        'trestbps': 'Resting blood pressure in mm Hg on admission to the hospital',
        'chol': 'Serum cholesterol in mg/dl',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
        'restecg': 'Resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes, 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of the peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping)',
        'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
        'thal': 'Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)',
        'num': 'Diagnosis of heart disease (0 = absence, 1-4 = presence with higher values indicating more severe condition)',
        'source': 'Source of the data (cleveland, hungarian, switzerland, va)',
        'target_binary': 'Binary target variable (0 = no disease, 1 = disease)'
    }

    return descriptions


def analyze_variable_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize variables by data type for analysis purposes.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze

    Returns
    -------
    Dict[str, List[str]]
        Dictionary with variable types as keys and lists of variable names as values
    """
    # Initialize categories
    categorical_vars = []
    binary_vars = []
    continuous_vars = []

    # Analyze each column
    for col in df.columns:
        # Skip the source column
        if col == 'source':
            continue

        # Check if binary (has only 0 and 1 values)
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            binary_vars.append(col)
        # Check if categorical (less than 10 unique values)
        elif len(unique_vals) < 10 and pd.api.types.is_numeric_dtype(df[col]):
            categorical_vars.append(col)
        # Otherwise treat as continuous
        elif pd.api.types.is_numeric_dtype(df[col]):
            continuous_vars.append(col)

    return {
        'binary': binary_vars,
        'categorical': categorical_vars,
        'continuous': continuous_vars
    }


def get_basic_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate basic statistics for all numeric variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze

    Returns
    -------
    Dict
        Dictionary containing statistical measures for each numeric variable
    """
    stats_dict = {}

    # Get variable types
    var_types = analyze_variable_types(df)
    numeric_vars = var_types['continuous'] + var_types['categorical'] + var_types['binary']

    # Calculate statistics for each numeric variable
    for var in numeric_vars:
        if var in df.columns and pd.api.types.is_numeric_dtype(df[var]):
            var_stats = {
                'mean': df[var].mean(),
                'median': df[var].median(),
                'std': df[var].std(),
                'min': df[var].min(),
                'max': df[var].max(),
                'q1': df[var].quantile(0.25),
                'q3': df[var].quantile(0.75),
                'iqr': df[var].quantile(0.75) - df[var].quantile(0.25),
                'missing': df[var].isna().sum(),
                'missing_pct': (df[var].isna().sum() / len(df)) * 100
            }
            stats_dict[var] = var_stats

    return stats_dict


def analyze_target_distribution(df: pd.DataFrame) -> Dict:
    """
    Analyze the distribution of the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with target variable

    Returns
    -------
    Dict
        Dictionary containing target distribution information
    """
    target_analysis = {}

    # Analyze original multi-class target
    if 'num' in df.columns:
        # Count of each class
        target_counts = df['num'].value_counts().sort_index().to_dict()
        target_analysis['original_target'] = {
            'counts': target_counts,
            'percentages': (df['num'].value_counts(normalize=True) * 100).sort_index().to_dict()
        }

    # Analyze binary target if available
    if 'target_binary' in df.columns:
        binary_counts = df['target_binary'].value_counts().sort_index().to_dict()
        target_analysis['binary_target'] = {
            'counts': binary_counts,
            'percentages': (df['target_binary'].value_counts(normalize=True) * 100).sort_index().to_dict()
        }

    return target_analysis


def analyze_gender_differences(df: pd.DataFrame) -> Dict:
    """
    Analyze differences between gender groups in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'sex' variable

    Returns
    -------
    Dict
        Dictionary containing gender-based analysis results
    """
    gender_analysis = {}

    if 'sex' not in df.columns:
        return {'error': 'Sex variable not found in dataset'}

    # Get counts by gender
    gender_counts = df['sex'].map({1: 'Male', 0: 'Female'}).value_counts().to_dict()
    gender_analysis['counts'] = gender_counts
    gender_analysis['percentages'] = (
                df['sex'].map({1: 'Male', 0: 'Female'}).value_counts(normalize=True) * 100).to_dict()

    # Analyze target variable by gender
    if 'target_binary' in df.columns:
        # Calculate disease prevalence by gender
        male_prevalence = df[df['sex'] == 1]['target_binary'].mean() * 100
        female_prevalence = df[df['sex'] == 0]['target_binary'].mean() * 100

        gender_analysis['disease_prevalence'] = {
            'Male': male_prevalence,
            'Female': female_prevalence
        }

    # Analyze key variables by gender
    var_types = analyze_variable_types(df)
    continuous_vars = var_types['continuous']

    gender_stats = {}
    for var in continuous_vars:
        if var in df.columns:
            male_stats = df[df['sex'] == 1][var].describe().to_dict()
            female_stats = df[df['sex'] == 0][var].describe().to_dict()

            # T-test for gender differences
            male_data = df[df['sex'] == 1][var].dropna()
            female_data = df[df['sex'] == 0][var].dropna()

            if len(male_data) > 0 and len(female_data) > 0:
                t_stat, p_value = stats.ttest_ind(male_data, female_data, equal_var=False, nan_policy='omit')

                gender_stats[var] = {
                    'male': male_stats,
                    'female': female_stats,
                    't_test': {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant_difference': p_value < 0.05
                    }
                }

    gender_analysis['variable_comparisons'] = gender_stats

    return gender_analysis


def calculate_correlations(df: pd.DataFrame) -> Dict:
    """
    Calculate correlations between numeric variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with numeric variables

    Returns
    -------
    Dict
        Dictionary containing correlation matrices
    """
    correlation_data = {}

    # Filter numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    # Calculate Pearson correlation
    try:
        pearson_corr = numeric_df.corr(method='pearson')
        correlation_data['pearson'] = pearson_corr.to_dict()
    except Exception as e:
        correlation_data['pearson_error'] = str(e)

    # Calculate Spearman correlation
    try:
        spearman_corr = numeric_df.corr(method='spearman')
        correlation_data['spearman'] = spearman_corr.to_dict()
    except Exception as e:
        correlation_data['spearman_error'] = str(e)

    return correlation_data


def perform_initial_analysis(df: pd.DataFrame) -> Dict:
    """
    Perform initial comprehensive analysis of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze

    Returns
    -------
    Dict
        Dictionary containing all analysis results
    """
    analysis_results = {}

    # Get variable descriptions
    analysis_results['variable_descriptions'] = get_variable_descriptions()

    # Categorize variables
    analysis_results['variable_types'] = analyze_variable_types(df)

    # Basic statistics
    analysis_results['basic_statistics'] = get_basic_statistics(df)

    # Target distribution
    analysis_results['target_analysis'] = analyze_target_distribution(df)

    # Gender differences
    analysis_results['gender_analysis'] = analyze_gender_differences(df)

    # Correlations
    analysis_results['correlations'] = calculate_correlations(df)

    return analysis_results


if __name__ == "__main__":
    # Test the functions if run directly
    print("This module provides analysis functions for the Heart Disease dataset.")
    print("Import and use in another script or notebook.")