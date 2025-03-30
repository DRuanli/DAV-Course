"""
Data Preprocessing Module for Heart Disease Dataset Analysis

This module implements a comprehensive data cleaning pipeline for the UCI
Heart Disease dataset, handling missing values, outliers, and feature engineering.

Author: Team Member 1
Date: March 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset with appropriate strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with missing values

    Returns
    -------
    pd.DataFrame
        Dataset with handled missing values

    Notes
    -----
    - Missing values in 'ca' and 'thal' are extensive and handled specially
    - Missing values in other columns use median/mode imputation
    - Physiologically impossible values (e.g., 0 for cholesterol) are treated as missing
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Replace '?' with NaN
    processed_df.replace('?', np.nan, inplace=True)

    # Handle physiologically impossible values
    processed_df.loc[processed_df['chol'] == 0, 'chol'] = np.nan
    processed_df.loc[processed_df['trestbps'] == 0, 'trestbps'] = np.nan

    # Calculate missing percentage for each column
    missing_pct = (processed_df.isnull().sum() / len(processed_df) * 100).round(2)
    print(f"Missing percentages before imputation:\n{missing_pct[missing_pct > 0]}")

    # For columns with high missing rates (> 50%), create a binary indicator and leave missing
    high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
    for col in high_missing_cols:
        processed_df[f'{col}_missing'] = processed_df[col].isnull().astype(int)

    # For continuous variables with moderate missing rates, impute with median
    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for col in continuous_cols:
        if col not in high_missing_cols and missing_pct[col] > 0:
            median_val = processed_df[col].median()
            processed_df[col].fillna(median_val, inplace=True)
            print(f"Imputed {col} missing values with median: {median_val}")

    # For categorical variables with moderate missing rates, impute with mode
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
    for col in categorical_cols:
        if col not in high_missing_cols and missing_pct[col] > 0:
            mode_val = processed_df[col].mode()[0]
            processed_df[col].fillna(mode_val, inplace=True)
            print(f"Imputed {col} missing values with mode: {mode_val}")

    # For 'ca' and 'thal' (likely high missing), use a missing indicator approach
    # if they have moderate missing rates
    if 'ca' not in high_missing_cols and 'ca' in processed_df.columns:
        mode_val = processed_df['ca'].mode()[0]
        processed_df['ca_missing'] = processed_df['ca'].isnull().astype(int)
        processed_df['ca'].fillna(mode_val, inplace=True)
        print(f"Created ca_missing indicator and imputed with mode: {mode_val}")

    if 'thal' not in high_missing_cols and 'thal' in processed_df.columns:
        mode_val = processed_df['thal'].mode()[0]
        processed_df['thal_missing'] = processed_df['thal'].isnull().astype(int)
        processed_df['thal'].fillna(mode_val, inplace=True)
        print(f"Created thal_missing indicator and imputed with mode: {mode_val}")

    # Calculate missing percentage after imputation
    missing_pct_after = (processed_df.isnull().sum() / len(processed_df) * 100).round(2)
    print(f"\nMissing percentages after imputation:\n{missing_pct_after[missing_pct_after > 0]}")

    return processed_df


def detect_and_handle_outliers(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Detect and handle outliers in continuous variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to check for outliers
    columns : List[str], optional
        Specific columns to check for outliers. If None, checks all numeric columns.

    Returns
    -------
    pd.DataFrame
        Dataset with handled outliers

    Notes
    -----
    - Uses IQR method to identify outliers
    - Creates indicators for outliers
    - Caps outliers at Q1-1.5*IQR and Q3+1.5*IQR
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # If no columns specified, use all numeric columns
    if columns is None:
        columns = processed_df.select_dtypes(include=['number']).columns.tolist()
        # Exclude binary columns and target
        columns = [col for col in columns if processed_df[col].nunique() > 2 and col != 'num'
                   and col != 'target_binary' and 'missing' not in col]

    outlier_summary = {}

    for col in columns:
        # Skip columns with too many missing values
        if processed_df[col].isnull().sum() / len(processed_df) > 0.5:
            continue

        # Calculate IQR
        Q1 = processed_df[col].quantile(0.25)
        Q3 = processed_df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Create outlier indicator
        processed_df[f'{col}_outlier'] = ((processed_df[col] < lower_bound) |
                                          (processed_df[col] > upper_bound)).astype(int)

        # Count outliers
        n_outliers = processed_df[f'{col}_outlier'].sum()
        outlier_pct = (n_outliers / len(processed_df) * 100).round(2)

        # Store outlier information
        outlier_summary[col] = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_outliers': n_outliers,
            'outlier_pct': outlier_pct
        }

        # Cap outliers
        processed_df[f'{col}_original'] = processed_df[col].copy()
        processed_df.loc[processed_df[col] < lower_bound, col] = lower_bound
        processed_df.loc[processed_df[col] > upper_bound, col] = upper_bound

    # Print outlier summary
    print("\nOutlier Summary:")
    for col, summary in outlier_summary.items():
        print(f"{col}: {summary['n_outliers']} outliers ({summary['outlier_pct']}%)")
        print(f"  Capped at: [{summary['lower_bound']:.2f}, {summary['upper_bound']:.2f}]")

    return processed_df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features to enhance the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset

    Returns
    -------
    pd.DataFrame
        Dataset with additional derived features
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Create age groups
    age_bins = [0, 40, 50, 60, 70, 100]
    age_labels = ['<40', '40-49', '50-59', '60-69', '70+']
    processed_df['age_group'] = pd.cut(processed_df['age'], bins=age_bins, labels=age_labels)

    # Create systolic blood pressure categories
    bp_bins = [0, 120, 140, 160, 300]
    bp_labels = ['normal', 'prehypertension', 'stage1', 'stage2']
    processed_df['bp_category'] = pd.cut(processed_df['trestbps'], bins=bp_bins, labels=bp_labels)

    # Create cholesterol categories based on clinical guidelines
    chol_bins = [0, 200, 240, 1000]
    chol_labels = ['desirable', 'borderline', 'high']
    processed_df['chol_category'] = pd.cut(processed_df['chol'], bins=chol_bins, labels=chol_labels)

    # Create ST depression categories
    oldpeak_bins = [-10, 0, 1.5, 10]
    oldpeak_labels = ['negative', 'minor', 'major']
    processed_df['oldpeak_category'] = pd.cut(processed_df['oldpeak'], bins=oldpeak_bins, labels=oldpeak_labels)

    # Create maximum heart rate relative to age
    # The formula 220 - age is a common approximation for maximum heart rate
    processed_df['max_hr_predicted'] = 220 - processed_df['age']
    processed_df['max_hr_pct'] = (processed_df['thalach'] / processed_df['max_hr_predicted'] * 100).round(1)

    # Create rate pressure product (RPP) - indicator of cardiac workload
    # RPP = systolic blood pressure * heart rate
    processed_df['rate_pressure_product'] = processed_df['trestbps'] * processed_df['thalach']

    # Print summary of derived features
    print("\nDerived Features Created:")
    print("- age_group: Age categories")
    print("- bp_category: Blood pressure categories")
    print("- chol_category: Cholesterol level categories")
    print("- oldpeak_category: ST depression severity")
    print("- max_hr_predicted: Predicted maximum heart rate based on age")
    print("- max_hr_pct: Percentage of achieved vs. predicted maximum heart rate")
    print("- rate_pressure_product: Cardiac workload indicator")

    return processed_df


def preprocess_heart_disease_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for the heart disease dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw heart disease dataset

    Returns
    -------
    pd.DataFrame
        Fully preprocessed dataset ready for analysis
    """
    print("Starting preprocessing pipeline...")

    # Handle missing values
    print("\n1. Handling missing values...")
    df_cleaned = handle_missing_values(df)

    # Detect and handle outliers
    print("\n2. Detecting and handling outliers...")
    df_cleaned = detect_and_handle_outliers(df_cleaned)

    # Create derived features
    print("\n3. Creating derived features...")
    df_processed = create_derived_features(df_cleaned)

    print("\nPreprocessing complete.")
    print(f"Original shape: {df.shape}")
    print(f"Processed shape: {df_processed.shape}")

    return df_processed


def document_preprocessing_decisions(df_original: pd.DataFrame, df_processed: pd.DataFrame) -> Dict:
    """
    Document all preprocessing decisions and their impact.

    Parameters
    ----------
    df_original : pd.DataFrame
        Original dataset before preprocessing
    df_processed : pd.DataFrame
        Dataset after preprocessing

    Returns
    -------
    Dict
        Dictionary documenting preprocessing decisions and their impact
    """
    documentation = {
        'original_shape': df_original.shape,
        'processed_shape': df_processed.shape,
        'new_columns': [col for col in df_processed.columns if col not in df_original.columns],
        'missing_values': {
            'before': df_original.isnull().sum().to_dict(),
            'after': df_processed.isnull().sum().to_dict(),
        },
        'column_data_types': {
            'before': df_original.dtypes.astype(str).to_dict(),
            'after': df_processed.dtypes.astype(str).to_dict(),
        },
        'summary_statistics': {
            'before': df_original.describe().to_dict(),
            'after': df_processed.describe().to_dict(),
        }
    }

    # Generate text summary
    print("\nPreprocessing Documentation Summary:")
    print(
        f"- Original dataset: {documentation['original_shape'][0]} rows, {documentation['original_shape'][1]} columns")
    print(
        f"- Processed dataset: {documentation['processed_shape'][0]} rows, {documentation['processed_shape'][1]} columns")
    print(f"- {len(documentation['new_columns'])} new columns created")

    print("\nMissing Values Impact:")
    for col in df_original.columns:
        before = documentation['missing_values']['before'].get(col, 0)
        after = documentation['missing_values']['after'].get(col, 0)
        if before > 0:
            print(f"- {col}: {before} missing values before → {after} after")

    return documentation


if __name__ == "__main__":
    # Test the module if run directly
    print("Heart Disease Dataset Preprocessing Module")
    print("This module provides preprocessing functions for the Heart Disease dataset.")
    print("Import and use in another script or notebook.")