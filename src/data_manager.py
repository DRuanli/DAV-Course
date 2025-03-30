"""
Data Manager Module for Heart Disease Dataset Analysis

This module handles loading, combining, and preprocessing data from all four UCI
Heart Disease dataset sources (Cleveland, Hungarian, Switzerland, VA).

Author: Team Member 1
Date: March 2025
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a single dataset file.

    Parameters
    ----------
    filepath : str
        Path to the dataset file

    Returns
    -------
    pd.DataFrame
        Loaded dataset with proper column names
    """
    # Column names for the heart disease dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]

    try:
        # Load data - handle different potential formats
        if filepath.endswith('.data'):
            df = pd.read_csv(filepath, header=None, names=column_names)
        else:
            df = pd.read_csv(filepath, header=None, names=column_names)

        return df
    except Exception as e:
        print(f"Error loading dataset {filepath}: {e}")
        return pd.DataFrame()


def combine_datasets() -> pd.DataFrame:
    """
    Load and combine all four heart disease dataset sources.

    Returns
    -------
    pd.DataFrame
        Combined dataset with source information
    """
    # Define dataset paths
    datasets = {
        'cleveland': 'data/processed.cleveland.data',
        'hungarian': 'data/processed.hungarian.data',
        'switzerland': 'data/processed.switzerland.data',
        'va': 'data/processed.va.data'
    }

    combined_df = pd.DataFrame()

    # Load and combine each dataset with source information
    for source, path in datasets.items():
        if os.path.exists(path):
            df = load_dataset(path)
            if not df.empty:
                df['source'] = source  # Add source information
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            print(f"Warning: Dataset file {path} not found")

    return combined_df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial preprocessing on the combined dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Combined raw dataset

    Returns
    -------
    pd.DataFrame
        Preprocessed dataset ready for analysis
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Replace '?' values with NaN
    processed_df.replace('?', np.nan, inplace=True)

    # Convert columns to appropriate types
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for col in numeric_cols:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

    # Convert categorical variables
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'num']
    for col in categorical_cols:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

    # Handle physiologically impossible values (e.g., cholesterol or blood pressure = 0)
    if 'chol' in processed_df.columns:
        processed_df.loc[processed_df['chol'] == 0, 'chol'] = np.nan

    if 'trestbps' in processed_df.columns:
        processed_df.loc[processed_df['trestbps'] == 0, 'trestbps'] = np.nan

    return processed_df


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate a summary of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to summarize

    Returns
    -------
    Dict
        Dictionary containing dataset summary information
    """
    summary = {
        'total_samples': len(df),
        'sources': df['source'].value_counts().to_dict() if 'source' in df.columns else {},
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'numeric_summary': df.describe().to_dict() if not df.empty else {}
    }

    return summary


def load_and_prepare_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to load, combine, preprocess datasets and generate summary.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Tuple containing preprocessed DataFrame and summary dictionary
    """
    # Combine all datasets
    combined_df = combine_datasets()

    # Apply preprocessing
    processed_df = preprocess_data(combined_df)

    # Generate summary
    summary = get_data_summary(processed_df)

    return processed_df, summary


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target variable from the multi-class 'num' variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the original 'num' column

    Returns
    -------
    pd.DataFrame
        DataFrame with additional binary target column
    """
    df_copy = df.copy()
    # Create binary target: 0 = no disease, 1 = disease (original values 1-4)
    df_copy['target_binary'] = df_copy['num'].apply(lambda x: 0 if x == 0 else 1)
    return df_copy


if __name__ == "__main__":
    # Test the functions if run directly
    print("Testing data loading and processing functions...")
    df, summary = load_and_prepare_data()
    print(f"Loaded dataset with {summary['total_samples']} samples")
    print(f"Missing values: {summary['missing_percentage']}")