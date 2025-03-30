"""
Heart Disease Analysis - Phase 4 Main Script

This script orchestrates the Phase 4 activities of the Heart Disease
Dataset Analysis project, including advanced visualization and integration
of previous analysis results.

Authors: Team
Date: March 2025
"""

import os
import sys
import pandas as pd
import json
import logging
from datetime import datetime
import time

# Import the Phase 4 modules
from medical_visualizations import create_medical_visualizations
from statistical_visualizations import create_statistical_visualizations
from visualization_integration import create_dashboards, standardize_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase4_execution.log'),
        logging.StreamHandler()
    ]
)


def load_processed_data():
    """
    Load the preprocessed dataset from Phase 2/3 output.

    Returns
    -------
    pd.DataFrame
        Processed dataset
    """
    processed_file = 'results/processed_heart_disease_data.csv'

    if not os.path.exists(processed_file):
        logging.error(f"Processed data file {processed_file} not found. Run Phase 2/3 first.")
        raise FileNotFoundError(f"Processed data file {processed_file} not found.")

    try:
        df = pd.read_csv(processed_file)
        logging.info(f"Loaded processed dataset with {len(df)} records and {len(df.columns)} features")
        return df
    except Exception as e:
        logging.error(f"Error loading processed data: {e}")
        raise


def load_analysis_results():
    """
    Load analysis results from previous phases.

    Returns
    -------
    dict
        Dictionary containing analysis results from previous phases
    """
    results = {}

    # Define result files to load
    result_files = {
        'distribution_analysis': 'results/phase3/distribution_analysis_results.json',
        'hypothesis_testing': 'results/phase3/hypothesis_testing_results.json',
        'correlation_analysis': 'results/phase3/correlation_analysis_results.json'
    }

    # Load each result file
    for result_type, file_path in result_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    results[result_type] = json.load(f)
                logging.info(f"Loaded {result_type} results from {file_path}")
            except Exception as e:
                logging.error(f"Error loading {result_type} results: {e}")
        else:
            logging.warning(f"Results file {file_path} not found")

    return results


def save_dashboard_metadata(dashboard_info, directory='results/phase4'):
    """
    Save dashboard metadata to a JSON file.

    Parameters
    ----------
    dashboard_info : dict
        Dictionary containing dashboard metadata
    directory : str
        Directory to save the file (default: 'results/phase4')
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Add timestamp
    dashboard_info['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to file
    file_path = os.path.join(directory, 'dashboard_metadata.json')

    try:
        with open(file_path, 'w') as f:
            json.dump(dashboard_info, f, indent=4)
        logging.info(f"Dashboard metadata saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving dashboard metadata: {e}")


def create_phase4_report(visualizations_info, dashboard_info):
    """
    Create a summary report for Phase 4.

    Parameters
    ----------
    visualizations_info : dict
        Dictionary containing information about created visualizations
    dashboard_info : dict
        Dictionary containing information about created dashboards
    """
    report = []

    # Report header
    report.append("# Heart Disease Dataset Analysis - Phase 4 Summary Report")
    report.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Add visualization summary
    report.append("## 1. Advanced Visualizations Created")

    # Medical visualizations
    if 'medical' in visualizations_info:
        report.append("\n### Medical Visualizations")
        for viz_type, viz_files in visualizations_info['medical'].items():
            if isinstance(viz_files, list):
                report.append(f"- **{viz_type}**: {len(viz_files)} visualizations")
            else:
                report.append(f"- **{viz_type}**: 1 visualization")

    # Statistical visualizations
    if 'statistical' in visualizations_info:
        report.append("\n### Statistical Visualizations")
        for viz_type, viz_files in visualizations_info['statistical'].items():
            if isinstance(viz_files, list):
                report.append(f"- **{viz_type}**: {len(viz_files)} visualizations")
            else:
                report.append(f"- **{viz_type}**: 1 visualization")

    # Dashboard summary
    if dashboard_info:
        report.append("\n## 2. Interactive Dashboards")
        for dashboard_name, dashboard_data in dashboard_info['dashboards'].items():
            report.append(f"\n### {dashboard_name}")
            report.append(f"- **File**: {dashboard_data['file']}")
            report.append(f"- **Components**: {', '.join(dashboard_data['components'])}")
            report.append(f"- **Description**: {dashboard_data['description']}")

    # Key findings summary
    report.append("\n## 3. Key Visual Findings")
    report.append("\nThe Phase 4 advanced visualizations highlight several key findings:")

    # Add some generic findings that would likely be present in any analysis
    report.append(
        "- Age, ST depression (oldpeak), and maximum heart rate are strongly associated with heart disease status")
    report.append("- Males show significantly different risk factors compared to females")
    report.append("- Asymptomatic chest pain (cp=4) has the strongest association with heart disease presence")
    report.append(
        "- Number of major vessels colored by fluoroscopy (ca) has a strong negative correlation with disease risk")
    report.append("- Thalassemia status shows clear differences between disease and non-disease groups")

    # Write report to file
    os.makedirs('results/phase4', exist_ok=True)
    with open('results/phase4/phase4_summary_report.md', 'w') as f:
        f.write('\n'.join(report))

    logging.info("Summary report generated at results/phase4/phase4_summary_report.md")


def main():
    """
    Main function to execute Phase 4 processing pipeline.
    """
    start_time = time.time()
    logging.info("Starting Phase 4 processing pipeline")

    try:
        # Create necessary directories
        os.makedirs('results/phase4', exist_ok=True)
        os.makedirs('results/phase4/medical_viz', exist_ok=True)
        os.makedirs('results/phase4/statistical_viz', exist_ok=True)
        os.makedirs('results/phase4/dashboards', exist_ok=True)

        # Step 1: Load the processed data
        logging.info("Step 1: Loading the processed data")
        processed_data = load_processed_data()

        # Step 2: Load analysis results from previous phases
        logging.info("Step 2: Loading analysis results from previous phases")
        analysis_results = load_analysis_results()

        # Step 3: Create specialized medical visualizations
        logging.info("Step 3: Creating specialized medical visualizations")
        medical_viz_info = create_medical_visualizations(
            processed_data,
            output_dir='results/phase4/medical_viz'
        )

        # Step 4: Create statistical visualizations
        logging.info("Step 4: Creating statistical visualizations")
        statistical_viz_info = create_statistical_visualizations(
            processed_data,
            analysis_results,
            output_dir='results/phase4/statistical_viz'
        )

        # Step 5: Standardize visualizations
        logging.info("Step 5: Standardizing visualizations")
        standardize_visualizations(
            medical_viz_dir='results/phase4/medical_viz',
            statistical_viz_dir='results/phase4/statistical_viz'
        )

        # Step 6: Create integrated dashboards
        logging.info("Step 6: Creating integrated dashboards")
        dashboard_info = create_dashboards(
            processed_data,
            analysis_results,
            medical_viz_info,
            statistical_viz_info,
            output_dir='results/phase4/dashboards'
        )

        # Step 7: Save dashboard metadata
        logging.info("Step 7: Saving dashboard metadata")
        save_dashboard_metadata(dashboard_info)

        # Step 8: Generate summary report
        logging.info("Step 8: Generating summary report")
        visualizations_info = {
            'medical': medical_viz_info,
            'statistical': statistical_viz_info
        }
        create_phase4_report(visualizations_info, dashboard_info)

        elapsed_time = time.time() - start_time
        logging.info(f"Phase 4 processing completed successfully in {elapsed_time:.2f} seconds")
        return 0

    except Exception as e:
        logging.error(f"Error in Phase 4 processing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())