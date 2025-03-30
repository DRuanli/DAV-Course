"""
Heart Disease Analysis - Phase 3 Main Script

This script orchestrates the Phase 3 activities of the Heart Disease
Dataset Analysis project, including advanced statistical analysis:
- Probability Distribution Analysis
- Hypothesis Testing
- Correlation and Relationship Analysis

Authors: Team
Date: March 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import time

# Import the Phase 3 modules
from distribution_analysis import perform_distribution_analysis
from hypothesis_testing import perform_hypothesis_testing
from correlation_analysis import perform_correlation_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase3_execution.log'),
        logging.StreamHandler()
    ]
)


def load_processed_data():
    """
    Load the preprocessed dataset from Phase 2 output.

    Returns
    -------
    pd.DataFrame
        Processed dataset
    """
    processed_file = 'results/processed_heart_disease_data.csv'

    if not os.path.exists(processed_file):
        logging.error(f"Processed data file {processed_file} not found. Run Phase 2 first.")
        raise FileNotFoundError(f"Processed data file {processed_file} not found.")

    try:
        df = pd.read_csv(processed_file)
        logging.info(f"Loaded processed dataset with {len(df)} records and {len(df.columns)} features")
        return df
    except Exception as e:
        logging.error(f"Error loading processed data: {e}")
        raise


def save_results(results, filename, directory='results/phase3'):
    """
    Save analysis results to a JSON file.

    Parameters
    ----------
    results : dict
        Analysis results to save
    filename : str
        Name of the file to save
    directory : str
        Directory to save the file (default: 'results/phase3')
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Add timestamp
    results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to file
    file_path = os.path.join(directory, filename)

    class NumpyEncoder(json.JSONEncoder):
        """Custom JSON encoder for NumPy types and Pandas objects."""

        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            return super(NumpyEncoder, self).default(obj)

    try:
        with open(file_path, 'w') as f:
            json.dump(results, f, cls=NumpyEncoder, indent=4)
        logging.info(f"Results saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving results to {file_path}: {e}")


def generate_summary_report(distribution_results, hypothesis_results, correlation_results):
    """
    Generate a summary report of Phase 3 analysis.

    Parameters
    ----------
    distribution_results : dict
        Results from distribution analysis
    hypothesis_results : dict
        Results from hypothesis testing
    correlation_results : dict
        Results from correlation analysis
    """
    report = []

    # Basic report header
    report.append("# Heart Disease Dataset Analysis - Phase 3 Summary Report")
    report.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Distribution Analysis section
    report.append("## 1. Probability Distribution Analysis")

    # Include key findings from distribution analysis
    if 'key_findings' in distribution_results:
        report.append("### Key Findings:")
        for finding in distribution_results['key_findings']:
            report.append(f"- {finding}")

    # Include normality test results
    if 'normality_tests' in distribution_results:
        report.append("\n### Normality Test Results:")
        for var, tests in distribution_results['normality_tests'].items():
            report.append(f"- **{var}**: ")
            if 'shapiro' in tests:
                p_value = tests['shapiro'].get('p_value', 0)
                result = "Normal" if p_value > 0.05 else "Non-normal"
                report.append(f"  - Shapiro-Wilk Test: {result} (p={p_value:.4f})")
            if 'ks' in tests:
                p_value = tests['ks'].get('p_value', 0)
                result = "Normal" if p_value > 0.05 else "Non-normal"
                report.append(f"  - Kolmogorov-Smirnov Test: {result} (p={p_value:.4f})")

    # Hypothesis Testing section
    report.append("\n## 2. Hypothesis Testing")

    # Include hypothesis test results
    if 'test_results' in hypothesis_results:
        report.append("### Hypothesis Test Results:")
        for test_name, test_result in hypothesis_results['test_results'].items():
            report.append(f"\n#### {test_name}")
            report.append(f"- Null Hypothesis: {test_result.get('null_hypothesis', 'N/A')}")
            report.append(f"- Test Statistic: {test_result.get('statistic', 'N/A')}")
            report.append(f"- P-value: {test_result.get('p_value', 'N/A')}")
            report.append(f"- Result: {test_result.get('conclusion', 'N/A')}")
            if 'effect_size' in test_result:
                report.append(f"- Effect Size: {test_result['effect_size'].get('value', 'N/A')} " +
                              f"({test_result['effect_size'].get('interpretation', 'N/A')})")

    # Correlation Analysis section
    report.append("\n## 3. Correlation and Relationship Analysis")

    # Include key correlations
    if 'key_correlations' in correlation_results:
        report.append("### Strongest Correlations:")
        for corr in correlation_results['key_correlations'][:10]:  # Top 10 correlations
            report.append(f"- {corr['var1']} and {corr['var2']}: r = {corr['value']:.3f} " +
                          f"(p = {corr['p_value']:.4f})")

    # Include partial correlation insights
    if 'partial_correlations' in correlation_results:
        report.append("\n### Key Partial Correlation Insights:")
        for insight in correlation_results.get('partial_correlation_insights', [])[:5]:
            report.append(f"- {insight}")

    # Conclusion section
    report.append("\n## 4. Conclusion and Clinical Interpretation")
    report.append("The Phase 3 analysis has provided several insights with potential clinical relevance:")

    # Combine key findings from all analyses
    all_key_findings = []
    all_key_findings.extend(distribution_results.get('clinical_interpretations', []))
    all_key_findings.extend(hypothesis_results.get('clinical_interpretations', []))
    all_key_findings.extend(correlation_results.get('clinical_interpretations', []))

    for finding in all_key_findings:
        report.append(f"- {finding}")

    # Write report to file
    os.makedirs('results/phase3', exist_ok=True)
    with open('results/phase3/phase3_summary_report.md', 'w') as f:
        f.write('\n'.join(report))

    logging.info("Summary report generated at results/phase3/phase3_summary_report.md")


def main():
    """
    Main function to execute Phase 3 processing pipeline.
    """
    start_time = time.time()
    logging.info("Starting Phase 3 processing pipeline")

    try:
        # Create necessary directories
        os.makedirs('results/phase3', exist_ok=True)
        os.makedirs('results/phase3/figures', exist_ok=True)

        # Step 1: Load the processed data
        logging.info("Step 1: Loading the processed data")
        processed_data = load_processed_data()

        # Step 2: Perform Probability Distribution Analysis
        logging.info("Step 2: Performing Probability Distribution Analysis")
        distribution_results = perform_distribution_analysis(processed_data)
        save_results(distribution_results, 'distribution_analysis_results.json')

        # Step 3: Perform Hypothesis Testing
        logging.info("Step 3: Performing Hypothesis Testing")
        hypothesis_results = perform_hypothesis_testing(processed_data)
        save_results(hypothesis_results, 'hypothesis_testing_results.json')

        # Step 4: Perform Correlation Analysis
        logging.info("Step 4: Performing Correlation Analysis")
        correlation_results = perform_correlation_analysis(processed_data)
        save_results(correlation_results, 'correlation_analysis_results.json')

        # Step 5: Generate summary report
        logging.info("Step 5: Generating summary report")
        generate_summary_report(distribution_results, hypothesis_results, correlation_results)

        elapsed_time = time.time() - start_time
        logging.info(f"Phase 3 processing completed successfully in {elapsed_time:.2f} seconds")
        return 0

    except Exception as e:
        logging.error(f"Error in Phase 3 processing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())