"""
Main execution script for Heart Disease Dataset Analysis

This script orchestrates the complete data analysis workflow for the
UCI Heart Disease dataset project.

Authors: Team
Date: March 2025
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import project modules
from src.data_manager import load_and_prepare_data, create_binary_target
from src.analysis import perform_initial_analysis, get_variable_descriptions
from src.visualization import setup_visualization_env, create_standard_visualizations
from src.utils import setup_logging, save_results, create_directory_structure, print_section_header, generate_html_report

# Set up argument parser
parser = argparse.ArgumentParser(description='Heart Disease Dataset Analysis')
parser.add_argument('--skip-visualizations', action='store_true', help='Skip creating visualizations')
parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
parser.add_argument('--log-file', type=str, default='heart_disease_analysis.log', help='Log file name')
args = parser.parse_args()


def initialize_project():
    """Initialize the project structure and environment."""
    # Set up logging
    setup_logging(args.log_file)
    logging.info("Starting Heart Disease Dataset Analysis")

    # Create directory structure
    create_directory_structure()
    logging.info("Project structure initialized")

    # Set up visualization environment
    setup_visualization_env()
    logging.info("Visualization environment set up")


def run_analysis_pipeline():
    """Run the complete analysis pipeline."""
    # Phase 1: Data Loading and Preprocessing
    print_section_header("Phase 1: Data Loading and Preprocessing")
    df, summary = load_and_prepare_data()

    # Add binary target variable
    df = create_binary_target(df)

    logging.info(f"Loaded dataset with {len(df)} samples from {df['source'].nunique()} sources")
    print(f"Dataset shape: {df.shape}")
    print(f"Sources: {', '.join(df['source'].unique())}")
    print(f"Missing values summary: {dict(df.isnull().sum().nlargest(5))}")

    # Phase 2: Initial Analysis
    print_section_header("Phase 2: Initial Analysis")
    analysis_results = perform_initial_analysis(df)

    # Save analysis results
    save_results(analysis_results, 'analysis_results.json', directory=args.output_dir)
    logging.info("Initial analysis completed and saved")

    # Display key findings
    target_analysis = analysis_results.get('target_analysis', {})
    if 'binary_target' in target_analysis:
        binary_counts = target_analysis['binary_target'].get('counts', {})
        binary_pcts = target_analysis['binary_target'].get('percentages', {})
        print("Disease Distribution:")
        print(f"  No Disease (0): {binary_counts.get('0', 0)} ({binary_pcts.get('0', 0):.2f}%)")
        print(f"  Disease (1): {binary_counts.get('1', 0)} ({binary_pcts.get('1', 0):.2f}%)")

    # Phase 3: Visualization
    if not args.skip_visualizations:
        print_section_header("Phase 3: Visualization")
        create_standard_visualizations(df, output_dir=os.path.join(args.output_dir, 'figures'))
        logging.info("Standard visualizations created")

    # Generate HTML report
    print_section_header("Generating Report")
    generate_html_report(df, analysis_results, output_file=os.path.join(args.output_dir, 'report.html'))
    logging.info("HTML report generated")

    print_section_header("Analysis Complete")
    print(f"Results saved to {args.output_dir}")
    print(f"See the HTML report at {os.path.join(args.output_dir, 'report.html')}")


def main():
    """Main function to execute the analysis pipeline."""
    try:
        # Initialize project
        initialize_project()

        # Run analysis pipeline
        run_analysis_pipeline()

        return 0
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())