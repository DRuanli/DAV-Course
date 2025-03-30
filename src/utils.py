"""
Utility Module for Heart Disease Dataset Analysis

This module provides utility functions for the UCI Heart Disease
dataset analysis project.

Author: Team
Date: March 2025
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import datetime
import logging


def setup_logging(log_file: str = 'heart_disease_analysis.log') -> None:
    """
    Set up logging configuration for the project.

    Parameters
    ----------
    log_file : str
        Name of the log file (default: 'heart_disease_analysis.log')
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('logs', log_file)),
            logging.StreamHandler()
        ]
    )

    logging.info("Logging initialized")


def save_results(results: Dict, filename: str, directory: str = 'results') -> None:
    """
    Save analysis results to a JSON file.

    Parameters
    ----------
    results : Dict
        Dictionary containing analysis results
    filename : str
        Name of the output file
    directory : str
        Directory to save the file (default: 'results')
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Add timestamp to results
    results['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

    with open(file_path, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=4)

    logging.info(f"Results saved to {file_path}")


def load_results(filename: str, directory: str = 'results') -> Dict:
    """
    Load analysis results from a JSON file.

    Parameters
    ----------
    filename : str
        Name of the file to load
    directory : str
        Directory containing the file (default: 'results')

    Returns
    -------
    Dict
        Dictionary containing the loaded results
    """
    file_path = os.path.join(directory, filename)

    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
        logging.info(f"Results loaded from {file_path}")
        return results
    except Exception as e:
        logging.error(f"Error loading results from {file_path}: {e}")
        return {}


def create_directory_structure() -> None:
    """
    Create the project directory structure.
    """
    directories = [
        'data',
        'notebooks',
        'src',
        'docs',
        'results',
        'results/figures',
        'logs'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    logging.info("Directory structure created")


def print_section_header(title: str, width: int = 80) -> None:
    """
    Print a formatted section header.

    Parameters
    ----------
    title : str
        Title of the section
    width : int
        Width of the header (default: 80)
    """
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width + "\n")


def generate_html_report(df: pd.DataFrame, analysis_results: Dict,
                         output_file: str = 'results/report.html') -> None:
    """
    Generate an HTML report with analysis results.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset used for analysis
    analysis_results : Dict
        Dictionary containing analysis results
    output_file : str
        Path for the output HTML file (default: 'results/report.html')
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Heart Disease Dataset Analysis</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                padding: 0;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #0066cc;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .summary {{
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 10px 0;
            }}
        </style>
    </head>
    <body>
        <h1>Heart Disease Dataset Analysis Report</h1>
        <p>Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="summary">
            <h2>Dataset Summary</h2>
            <p>Total samples: {analysis_results.get('total_samples', len(df))}</p>
            <p>Dataset sources: {', '.join(df['source'].unique()) if 'source' in df.columns else 'N/A'}</p>
            <p>Variables: {', '.join(df.columns)}</p>
        </div>

        <h2>Dataset Description</h2>
        <table>
            <tr>
                <th>Variable</th>
                <th>Description</th>
                <th>Type</th>
                <th>Missing Values</th>
            </tr>
    """

    # Add variable descriptions
    var_desc = analysis_results.get('variable_descriptions', {})
    for var in df.columns:
        html_content += f"""
            <tr>
                <td>{var}</td>
                <td>{var_desc.get(var, 'No description available')}</td>
                <td>{df[var].dtype}</td>
                <td>{df[var].isna().sum()} ({(df[var].isna().sum() / len(df) * 100):.2f}%)</td>
            </tr>
        """

    html_content += """
        </table>

        <h2>Analysis Results</h2>
    """

    # Add basic statistics
    basic_stats = analysis_results.get('basic_statistics', {})
    if basic_stats:
        html_content += """
        <h3>Basic Statistics</h3>
        <table>
            <tr>
                <th>Variable</th>
                <th>Mean</th>
                <th>Median</th>
                <th>Std</th>
                <th>Min</th>
                <th>Max</th>
            </tr>
        """

        for var, stats in basic_stats.items():
            html_content += f"""
            <tr>
                <td>{var}</td>
                <td>{stats.get('mean', 'N/A'):.2f}</td>
                <td>{stats.get('median', 'N/A'):.2f}</td>
                <td>{stats.get('std', 'N/A'):.2f}</td>
                <td>{stats.get('min', 'N/A'):.2f}</td>
                <td>{stats.get('max', 'N/A'):.2f}</td>
            </tr>
            """

        html_content += """
        </table>
        """

    # Add target distribution
    target_analysis = analysis_results.get('target_analysis', {})
    if target_analysis and 'original_target' in target_analysis:
        counts = target_analysis['original_target'].get('counts', {})
        percentages = target_analysis['original_target'].get('percentages', {})

        html_content += """
        <h3>Target Distribution</h3>
        <table>
            <tr>
                <th>Class</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
        """

        for cls in sorted(counts.keys()):
            html_content += f"""
            <tr>
                <td>{cls}</td>
                <td>{counts.get(cls, 0)}</td>
                <td>{percentages.get(cls, 0):.2f}%</td>
            </tr>
            """

        html_content += """
        </table>
        """

    # List available figures
    figures_dir = 'results/figures'
    if os.path.exists(figures_dir):
        figures = [f for f in os.listdir(figures_dir) if f.endswith('.png')]

        if figures:
            html_content += """
            <h2>Visualizations</h2>
            """

            for figure in figures:
                figure_path = f"figures/{figure}"
                html_content += f"""
                <h3>{figure.replace('_', ' ').replace('.png', '')}</h3>
                <img src="{figure_path}" alt="{figure}">
                """

    # Close HTML document
    html_content += """
    </body>
    </html>
    """

    # Write to file
    with open(output_file, 'w') as f:
        f.write(html_content)

    logging.info(f"HTML report generated at {output_file}")


if __name__ == "__main__":
    # Test the functions if run directly
    setup_logging()
    create_directory_structure()
    print("Utility functions are ready for use.")