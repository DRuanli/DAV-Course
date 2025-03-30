"""
Heart Disease Analysis - Phase 2 Main Script

This script orchestrates the Phase 2 activities of the Heart Disease
Dataset Analysis project, including data preprocessing, statistical
analysis, and visualization.

Authors: Team
Date: March 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import time
import logging
from datetime import datetime

# Import the Phase 2 modules
from data_preprocessing import preprocess_heart_disease_data, document_preprocessing_decisions
from statistical_analysis import perform_statistical_analysis
from visualization_suite import create_all_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase2_execution.log'),
        logging.StreamHandler()
    ]
)


def load_data():
    """
    Load and combine the UCI Heart Disease dataset from all four sources.

    Returns
    -------
    pd.DataFrame
        Combined dataset
    """
    # Define column names
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]

    # Define data sources
    data_sources = {
        'cleveland': 'data/processed.cleveland.data',
        'hungarian': 'data/processed.hungarian.data',
        'switzerland': 'data/processed.switzerland.data',
        'va': 'data/processed.va.data'
    }

    combined_df = pd.DataFrame()

    # Load and combine each dataset
    for source, path in data_sources.items():
        if os.path.exists(path):
            logging.info(f"Loading {source} dataset from {path}")
            try:
                df = pd.read_csv(path, header=None, names=column_names)
                df['source'] = source
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                logging.info(f"Added {len(df)} records from {source}")
            except Exception as e:
                logging.error(f"Error loading {source} dataset: {e}")
        else:
            logging.warning(f"Dataset file {path} not found")

    logging.info(f"Combined dataset has {len(combined_df)} records")
    return combined_df


def save_results(results, filename, directory='results'):
    """
    Save analysis results to a JSON file.

    Parameters
    ----------
    results : dict
        Analysis results to save
    filename : str
        Name of the file to save
    directory : str
        Directory to save the file (default: 'results')
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Add timestamp
    results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to file
    file_path = os.path.join(directory, filename)

    class NumpyEncoder(json.JSONEncoder):
        """Custom JSON encoder for NumPy types."""

        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return super(NumpyEncoder, self).default(obj)

    try:
        with open(file_path, 'w') as f:
            json.dump(results, f, cls=NumpyEncoder, indent=4)
        logging.info(f"Results saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving results to {file_path}: {e}")


def main():
    """
    Main function to execute Phase 2 processing pipeline.
    """
    start_time = time.time()
    logging.info("Starting Phase 2 processing pipeline")

    try:
        # Create necessary directories
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/figures', exist_ok=True)

        # Step 1: Load the raw data
        logging.info("Step 1: Loading the raw data")
        raw_data = load_data()

        # Step 2: Preprocess the data
        logging.info("Step 2: Preprocessing the data")
        processed_data = preprocess_heart_disease_data(raw_data)

        # Document preprocessing decisions
        preprocessing_doc = document_preprocessing_decisions(raw_data, processed_data)
        save_results(preprocessing_doc, 'preprocessing_documentation.json')

        # Step 3: Perform statistical analysis
        logging.info("Step 3: Performing statistical analysis")
        statistical_results = perform_statistical_analysis(processed_data)
        save_results(statistical_results, 'statistical_analysis_results.json')

        # Step 4: Create visualizations
        logging.info("Step 4: Creating visualizations")
        visualization_results = create_all_visualizations(processed_data)
        save_results({"visualization_files": visualization_results}, 'visualization_results.json')

        # Save processed dataset
        processed_data.to_csv('results/processed_heart_disease_data.csv', index=False)
        logging.info("Processed dataset saved to results/processed_heart_disease_data.csv")

        # Generate summary report
        logging.info("Step 5: Generating summary report")
        generate_summary_report(raw_data, processed_data, statistical_results, visualization_results)

        elapsed_time = time.time() - start_time
        logging.info(f"Phase 2 processing completed successfully in {elapsed_time:.2f} seconds")
        return 0

    except Exception as e:
        logging.error(f"Error in Phase 2 processing: {e}", exc_info=True)
        return 1


def generate_summary_report(raw_data, processed_data, statistical_results, visualization_results):
    """
    Generate a summary report of Phase 2 analysis.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Original dataset
    processed_data : pd.DataFrame
        Preprocessed dataset
    statistical_results : dict
        Results from statistical analysis
    visualization_results : dict
        Results from visualization
    """
    report = []

    # Basic dataset information
    report.append("# Heart Disease Dataset Analysis - Phase 2 Summary Report")
    report.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Data overview
    report.append("## 1. Dataset Overview")
    report.append(f"- Original dataset: {len(raw_data)} records, {len(raw_data.columns)} features")
    report.append(f"- Processed dataset: {len(processed_data)} records, {len(processed_data.columns)} features")
    report.append(f"- Data sources: {', '.join(raw_data['source'].unique())}")

    if 'target_binary' in processed_data.columns:
        disease_pct = processed_data['target_binary'].mean() * 100
        report.append(f"- Disease prevalence: {disease_pct:.1f}%")

    # Preprocessing summary
    report.append("\n## 2. Data Preprocessing Summary")

    # Missing values handling
    missing_before = raw_data.isnull().sum().sum()
    missing_after = processed_data.isnull().sum().sum()
    report.append(f"- Missing values: {missing_before} values before preprocessing, {missing_after} values after")

    # New features
    new_features = [col for col in processed_data.columns if col not in raw_data.columns]
    report.append(f"- New features created: {len(new_features)}")
    if new_features:
        report.append(f"  - {', '.join(new_features[:10])}" + ("..." if len(new_features) > 10 else ""))

    # Statistical highlights
    report.append("\n## 3. Key Statistical Findings")

    # Check if we have patterns in statistical results
    if 'patterns' in statistical_results and 'subgroup_insights' in statistical_results['patterns']:
        insights = statistical_results['patterns']['subgroup_insights']
        if insights:
            report.append("### Notable Patterns:")
            for insight in insights[:5]:  # Limit to top 5 insights
                report.append(f"- {insight}")

    # Check if we have disease analysis in statistical results
    if 'disease_analysis' in statistical_results and 'continuous_variables' in statistical_results['disease_analysis']:
        # Get top predictors
        cont_vars = statistical_results['disease_analysis']['continuous_variables']
        if isinstance(cont_vars, pd.DataFrame):
            top_vars = cont_vars.head(5)
            report.append("\n### Top Predictors of Heart Disease:")
            for idx, row in top_vars.iterrows():
                if 'effect_magnitude' in row and 'p_value' in row:
                    report.append(f"- {idx}: {row['effect_magnitude']} effect (p={row['p_value']:.4f})")

    # Visualization summary
    report.append("\n## 4. Visualization Summary")
    report.append(
        f"- Created {sum(len(files) for files in visualization_results.values() if isinstance(files, dict))} visualizations")
    report.append("- Key visualizations include:")
    report.append("  - Histograms of continuous variables")
    report.append("  - Bar charts of categorical variables")
    report.append("  - Correlation heatmap")
    report.append("  - Boxplots comparing features across disease status")

    # Areas for further investigation
    report.append("\n## 5. Areas for Further Investigation")

    if 'patterns' in statistical_results and 'investigation_areas' in statistical_results['patterns']:
        areas = statistical_results['patterns']['investigation_areas']
        if areas:
            for area in areas[:5]:  # Limit to top 5 areas
                report.append(f"- {area}")
    else:
        # Default investigations if statistical patterns not available
        report.append("- Explore non-linear relationships between variables")
        report.append("- Investigate interaction effects between risk factors")
        report.append("- Analyze gender-specific patterns in disease presentation")

    # Write report to file
    with open('results/phase2_summary_report.md', 'w') as f:
        f.write('\n'.join(report))

    logging.info("Summary report generated at results/phase2_summary_report.md")


if __name__ == "__main__":
    sys.exit(main())