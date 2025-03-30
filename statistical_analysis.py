"""
Statistical Analysis Module for Heart Disease Dataset Analysis

This module implements comprehensive statistical analysis for the UCI
Heart Disease dataset, including descriptive statistics, segmented analysis,
and pattern identification.

Author: Team Member 2
Date: March 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import scipy.stats as stats
from collections import OrderedDict


def calculate_descriptive_statistics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calculate comprehensive descriptive statistics for all variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing descriptive statistics for different variable types
    """
    results = {}

    # Identify variable types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Filter out certain columns
    numeric_cols = [col for col in numeric_cols if 'missing' not in col and 'outlier' not in col]

    # Continuous variables statistics
    continuous_stats = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    # Add additional statistics
    continuous_stats.loc['var'] = df[numeric_cols].var()
    continuous_stats.loc['skew'] = df[numeric_cols].skew()
    continuous_stats.loc['kurt'] = df[numeric_cols].kurtosis()
    continuous_stats.loc['missing'] = df[numeric_cols].isnull().sum()
    continuous_stats.loc['missing_pct'] = df[numeric_cols].isnull().sum() / len(df) * 100

    results['continuous'] = continuous_stats

    # Categorical variables statistics
    categorical_results = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts().to_dict()
        proportions = df[col].value_counts(normalize=True).mul(100).round(2).to_dict()

        # Combine counts and proportions
        cat_stats = OrderedDict()
        for value in sorted(value_counts.keys()):
            cat_stats[value] = {
                'count': value_counts[value],
                'percentage': proportions[value]
            }

        categorical_results[col] = pd.DataFrame.from_dict(cat_stats, orient='index')

    results['categorical'] = categorical_results

    # Binary variables statistics (treat as special case)
    binary_cols = [col for col in numeric_cols if set(df[col].dropna().unique()).issubset({0, 1})]
    if binary_cols:
        binary_stats = {}
        for col in binary_cols:
            counts = df[col].value_counts().to_dict()
            proportions = df[col].value_counts(normalize=True).mul(100).round(2).to_dict()
            # Make sure 0 and 1 are present in the results
            binary_stats[col] = {
                '0_count': counts.get(0, 0),
                '0_pct': proportions.get(0, 0),
                '1_count': counts.get(1, 0),
                '1_pct': proportions.get(1, 0)
            }

        results['binary'] = pd.DataFrame.from_dict(binary_stats, orient='index')

    return results


def segment_analysis_by_gender(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Perform segmented analysis by gender.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing 'sex' variable (0=female, 1=male)

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing statistical comparisons by gender
    """
    if 'sex' not in df.columns:
        return {'error': 'Sex variable not found in dataset'}

    # Define groups
    female_df = df[df['sex'] == 0]
    male_df = df[df['sex'] == 1]

    results = {}

    # Basic demographics
    demographics = {
        'count': [len(female_df), len(male_df)],
        'percentage': [len(female_df) / len(df) * 100, len(male_df) / len(df) * 100],
        'disease_count': [female_df['target_binary'].sum(), male_df['target_binary'].sum()],
        'disease_pct': [female_df['target_binary'].mean() * 100, male_df['target_binary'].mean() * 100]
    }

    results['demographics'] = pd.DataFrame(demographics, index=['Female', 'Male'])

    # Continuous variables comparison
    continuous_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Filter out binary, target, and derived columns
    continuous_cols = [col for col in continuous_cols if col not in ['sex', 'num', 'target_binary']
                      and 'missing' not in col and 'outlier' not in col and 'original' not in col]

    # Calculate statistics for each continuous variable
    variable_stats = {}
    for var in continuous_cols:
        # Calculate statistics
        female_stats = female_df[var].describe()
        male_stats = male_df[var].describe()

        # T-test for significance
        t_stat, p_value = stats.ttest_ind(
            female_df[var].dropna(),
            male_df[var].dropna(),
            equal_var=False,
            nan_policy='omit'
        )

        # Effect size (Cohen's d)
        female_mean = female_df[var].mean()
        male_mean = male_df[var].mean()
        female_std = female_df[var].std()
        male_std = male_df[var].std()

        # Pooled standard deviation
        n_female = female_df[var].count()
        n_male = male_df[var].count()
        pooled_std = np.sqrt(((n_female - 1) * female_std**2 + (n_male - 1) * male_std**2) /
                            (n_female + n_male - 2))

        cohens_d = abs(female_mean - male_mean) / pooled_std

        variable_stats[var] = {
            'female_mean': female_mean,
            'male_mean': male_mean,
            'difference': male_mean - female_mean,
            'difference_pct': (male_mean - female_mean) / female_mean * 100 if female_mean != 0 else np.nan,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': cohens_d,
            'effect_magnitude': 'Small' if cohens_d < 0.5 else 'Medium' if cohens_d < 0.8 else 'Large'
        }

    results['continuous_variables'] = pd.DataFrame.from_dict(variable_stats, orient='index')

    # Disease prevalence by gender and other categorical variables
    categorical_cols = [col for col in df.columns if col.endswith('_category') or col.endswith('_group')]

    categorical_comparisons = {}
    for cat_var in categorical_cols:
        if cat_var in df.columns:
            # Create contingency tables
            female_counts = pd.crosstab(
                female_df[cat_var],
                female_df['target_binary'],
                normalize='index'
            ).mul(100).round(2)

            male_counts = pd.crosstab(
                male_df[cat_var],
                male_df['target_binary'],
                normalize='index'
            ).mul(100).round(2)

            # Rename columns for clarity
            female_counts.columns = ['Female_No_Disease_%', 'Female_Disease_%']
            male_counts.columns = ['Male_No_Disease_%', 'Male_Disease_%']

            # Combine tables
            combined = pd.concat([female_counts, male_counts], axis=1)
            categorical_comparisons[cat_var] = combined

    results['categorical_comparisons'] = categorical_comparisons

    return results


def segment_analysis_by_age(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Perform segmented analysis by age groups.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing 'age' variable

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing statistical comparisons by age groups
    """
    if 'age' not in df.columns:
        return {'error': 'Age variable not found in dataset'}

    # Create age groups if they don't exist
    if 'age_group' not in df.columns:
        age_bins = [0, 40, 50, 60, 70, 100]
        age_labels = ['<40', '40-49', '50-59', '60-69', '70+']
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

    results = {}

    # Basic demographics by age group
    age_demographics = df.groupby('age_group').agg(
        count=('age', 'count'),
        percentage=('age', lambda x: len(x) / len(df) * 100),
        disease_count=('target_binary', 'sum'),
        disease_pct=('target_binary', lambda x: x.mean() * 100)
    ).round(2)

    results['demographics'] = age_demographics

    # Continuous variables by age group
    continuous_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Filter out binary, target, and derived columns
    continuous_cols = [col for col in continuous_cols if col not in ['age', 'num', 'target_binary']
                      and 'missing' not in col and 'outlier' not in col and 'original' not in col]

    # Calculate statistics for each continuous variable by age group
    continuous_by_age = {}
    for var in continuous_cols:
        # Calculate statistics
        stats_by_age = df.groupby('age_group')[var].agg(['mean', 'median', 'std']).round(2)
        continuous_by_age[var] = stats_by_age

    results['continuous_variables'] = continuous_by_age

    # ANOVA test for significant differences across age groups
    anova_results = {}
    for var in continuous_cols:
        # Group data by age group
        groups = []
        for age_group in df['age_group'].unique():
            group_data = df[df['age_group'] == age_group][var].dropna()
            if len(group_data) > 0:
                groups.append(group_data)

        if len(groups) > 1:
            # Perform ANOVA
            f_statistic, p_value = stats.f_oneway(*groups)

            # Calculate effect size (eta squared)
            group_means = [group.mean() for group in groups]
            group_counts = [len(group) for group in groups]
            grand_mean = df[var].mean()

            # Between-group sum of squares
            ss_between = sum(count * (mean - grand_mean)**2 for count, mean in zip(group_counts, group_means))

            # Total sum of squares
            ss_total = sum((df[var] - grand_mean)**2)

            # Eta squared
            eta_squared = ss_between / ss_total if ss_total != 0 else 0

            anova_results[var] = {
                'f_statistic': f_statistic,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'eta_squared': eta_squared,
                'effect_magnitude': 'Small' if eta_squared < 0.06 else 'Medium' if eta_squared < 0.14 else 'Large'
            }

    results['anova_tests'] = pd.DataFrame.from_dict(anova_results, orient='index')

    # Disease prevalence by age group and other categorical variables
    categorical_cols = [col for col in df.columns if col.endswith('_category') or '_group' in col]
    categorical_cols = [col for col in categorical_cols if col != 'age_group']

    disease_by_category_age = {}
    for cat_var in categorical_cols:
        if cat_var in df.columns:
            # Create pivot table
            pivot = pd.pivot_table(
                df,
                values='target_binary',
                index=[cat_var],
                columns=['age_group'],
                aggfunc=lambda x: x.mean() * 100
            ).round(2)

            # Add overall column
            pivot['Overall'] = df.groupby(cat_var)['target_binary'].mean() * 100

            disease_by_category_age[cat_var] = pivot

    results['disease_prevalence'] = disease_by_category_age

    return results


def segment_analysis_by_disease(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Perform segmented analysis by disease status.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing disease target variable

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing statistical comparisons by disease status
    """
    if 'target_binary' not in df.columns:
        if 'num' in df.columns:
            df['target_binary'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
        else:
            return {'error': 'Target variable not found in dataset'}

    # Define groups
    no_disease = df[df['target_binary'] == 0]
    disease = df[df['target_binary'] == 1]

    results = {}

    # Basic demographics
    demographics = {
        'count': [len(no_disease), len(disease)],
        'percentage': [len(no_disease) / len(df) * 100, len(disease) / len(df) * 100],
        'avg_age': [no_disease['age'].mean(), disease['age'].mean()],
        'male_pct': [no_disease['sex'].mean() * 100, disease['sex'].mean() * 100]
    }

    results['demographics'] = pd.DataFrame(demographics, index=['No Disease', 'Disease'])

    # Continuous variables comparison
    continuous_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Filter out binary, target, and derived columns
    continuous_cols = [col for col in continuous_cols if col not in ['num', 'target_binary']
                      and 'missing' not in col and 'outlier' not in col and 'original' not in col]

    # Calculate statistics for each continuous variable
    variable_stats = {}
    for var in continuous_cols:
        # Calculate statistics
        no_disease_stats = no_disease[var].describe()
        disease_stats = disease[var].describe()

        # T-test for significance
        t_stat, p_value = stats.ttest_ind(
            no_disease[var].dropna(),
            disease[var].dropna(),
            equal_var=False,
            nan_policy='omit'
        )

        # Effect size (Cohen's d)
        no_disease_mean = no_disease[var].mean()
        disease_mean = disease[var].mean()
        no_disease_std = no_disease[var].std()
        disease_std = disease[var].std()

        # Pooled standard deviation
        n_no_disease = no_disease[var].count()
        n_disease = disease[var].count()
        pooled_std = np.sqrt(((n_no_disease - 1) * no_disease_std**2 + (n_disease - 1) * disease_std**2) /
                            (n_no_disease + n_disease - 2))

        cohens_d = abs(disease_mean - no_disease_mean) / pooled_std

        variable_stats[var] = {
            'no_disease_mean': no_disease_mean,
            'disease_mean': disease_mean,
            'difference': disease_mean - no_disease_mean,
            'difference_pct': ((disease_mean - no_disease_mean) / no_disease_mean * 100) if no_disease_mean != 0 else np.nan,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': cohens_d,
            'effect_magnitude': 'Small' if cohens_d < 0.5 else 'Medium' if cohens_d < 0.8 else 'Large'
        }

    # Sort by effect size (strongest associations first)
    variable_stats = {k: v for k, v in sorted(variable_stats.items(),
                                              key=lambda item: abs(item[1]['effect_size']),
                                              reverse=True)}

    results['continuous_variables'] = pd.DataFrame.from_dict(variable_stats, orient='index')

    # Categorical variables comparison
    categorical_cols = [col for col in df.columns if col.endswith('_category') or col.endswith('_group')]
    categorical_stats = {}

    for cat_var in categorical_cols:
        if cat_var in df.columns:
            # Create contingency table
            contingency = pd.crosstab(df[cat_var], df['target_binary'])

            # Calculate chi-square
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

            # Calculate Cramer's V (effect size for chi-square)
            n = contingency.sum().sum()
            cramer_v = np.sqrt(chi2 / (n * min(contingency.shape) - 1)) if n > 0 else 0

            # Calculate percentages by group
            pct_table = pd.crosstab(
                df[cat_var],
                df['target_binary'],
                normalize='index'
            ).mul(100).round(2)

            # Rename columns for clarity
            pct_table.columns = ['No Disease %', 'Disease %']

            categorical_stats[cat_var] = {
                'contingency': contingency,
                'percentages': pct_table,
                'chi2': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'cramer_v': cramer_v,
                'effect_magnitude': 'Small' if cramer_v < 0.3 else 'Medium' if cramer_v < 0.5 else 'Large'
            }

    results['categorical_variables'] = categorical_stats

    # Risk factors analysis - calculating odds ratios for binary predictors
    binary_cols = [col for col in df.columns if set(df[col].dropna().unique()).issubset({0, 1})
                  and col not in ['num', 'target_binary']]

    risk_factors = {}
    for factor in binary_cols:
        # Create contingency table
        table = pd.crosstab(df[factor], df['target_binary'])

        # Make sure the table has the right format [0,0], [0,1], [1,0], [1,1]
        if 0 not in table.index or 1 not in table.index or 0 not in table.columns or 1 not in table.columns:
            continue

        # Calculate odds ratio
        odds_ratio = (table.loc[1, 1] * table.loc[0, 0]) / (table.loc[1, 0] * table.loc[0, 1])

        # Fisher's exact test
        oddsratio, p_value = stats.fisher_exact(table)

        risk_factors[factor] = {
            'odds_ratio': odds_ratio,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'risk_level': 'Protective' if odds_ratio < 1 else 'Neutral' if odds_ratio == 1 else 'Risk factor'
        }

    # Sort by odds ratio (strongest associations first)
    risk_factors = {k: v for k, v in sorted(risk_factors.items(),
                                           key=lambda item: abs(item[1]['odds_ratio']),
                                           reverse=True)}

    results['risk_factors'] = pd.DataFrame.from_dict(risk_factors, orient='index')

    return results


def identify_notable_patterns(df: pd.DataFrame) -> Dict:
    """
    Identify notable patterns and potential areas for deeper investigation.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset

    Returns
    -------
    Dict
        Dictionary of notable patterns and investigation areas
    """
    patterns = {
        'correlations': {},
        'subgroup_insights': [],
        'risk_profiles': [],
        'investigation_areas': []
    }

    # Strong correlations with target
    if 'target_binary' in df.columns:
        # Numeric correlations
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['num', 'target_binary']
                       and 'missing' not in col and 'outlier' not in col]

        correlations = {}
        for col in numeric_cols:
            corr = df[col].corr(df['target_binary'])
            correlations[col] = corr

        # Sort by absolute correlation value
        correlations = {k: v for k, v in sorted(correlations.items(),
                                               key=lambda item: abs(item[1]),
                                               reverse=True)}

        # Keep top correlations
        top_correlations = {k: v for k, v in list(correlations.items())[:10]}
        patterns['correlations'] = top_correlations

        # Identify strong predictors
        strong_predictors = [k for k, v in correlations.items() if abs(v) > 0.3]
        if strong_predictors:
            patterns['investigation_areas'].append(
                f"Strong predictors of heart disease: {', '.join(strong_predictors)}"
            )

    # Gender differences
    if 'sex' in df.columns and 'target_binary' in df.columns:
        male_prevalence = df[df['sex'] == 1]['target_binary'].mean() * 100
        female_prevalence = df[df['sex'] == 0]['target_binary'].mean() * 100

        if abs(male_prevalence - female_prevalence) > 10:
            patterns['subgroup_insights'].append(
                f"Notable gender difference in disease prevalence: Males {male_prevalence:.1f}% vs. Females {female_prevalence:.1f}%"
            )
            patterns['investigation_areas'].append(
                "Investigate gender-specific risk factors and diagnostic criteria"
            )

    # Age patterns
    if 'age' in df.columns and 'target_binary' in df.columns:
        # Calculate disease prevalence by age decade
        df['age_decade'] = (df['age'] // 10) * 10
        prevalence_by_decade = df.groupby('age_decade')['target_binary'].mean() * 100

        # Check for monotonic increase with age
        is_increasing = all(x <= y for x, y in zip(prevalence_by_decade, prevalence_by_decade[1:]))
        if is_increasing:
            patterns['subgroup_insights'].append(
                "Disease prevalence increases monotonically with age"
            )

        # High-risk age groups
        high_risk_decades = prevalence_by_decade[prevalence_by_decade > 60].index.tolist()
        if high_risk_decades:
            patterns['risk_profiles'].append(
                f"High-risk age groups: {', '.join([f'{int(decade)}s' for decade in high_risk_decades])}"
            )

        # Young patients with disease
        young_patients = df[(df['age'] < 45) & (df['target_binary'] == 1)]
        if len(young_patients) >= 5:
            patterns['subgroup_insights'].append(
                f"Notable presence of young patients with heart disease: {len(young_patients)} patients under 45"
            )
            patterns['investigation_areas'].append(
                "Investigate risk factors for early-onset heart disease"
            )

    # Symptom patterns
    if 'cp' in df.columns and 'target_binary' in df.columns:
        # Check for asymptomatic cases (cp=4)
        asymptomatic = df[df['cp'] == 4]
        if len(asymptomatic) > 0:
            asymptomatic_prevalence = asymptomatic['target_binary'].mean() * 100
            overall_prevalence = df['target_binary'].mean() * 100

            if asymptomatic_prevalence > overall_prevalence + 10:
                patterns['subgroup_insights'].append(
                    f"High disease prevalence in asymptomatic patients: {asymptomatic_prevalence:.1f}% vs. overall {overall_prevalence:.1f}%"
                )
                patterns['investigation_areas'].append(
                    "Investigate diagnostic methods for asymptomatic patients"
                )

    # Thalach (max heart rate) patterns
    if 'thalach' in df.columns and 'target_binary' in df.columns:
        low_thalach = df[df['thalach'] < df['thalach'].quantile(0.25)]
        if len(low_thalach) > 0:
            low_thalach_prevalence = low_thalach['target_binary'].mean() * 100
            overall_prevalence = df['target_binary'].mean() * 100

            if low_thalach_prevalence > overall_prevalence + 10:
                patterns['subgroup_insights'].append(
                    f"High disease prevalence in patients with low maximum heart rate: {low_thalach_prevalence:.1f}% vs. overall {overall_prevalence:.1f}%"
                )

    # ST depression patterns
    if 'oldpeak' in df.columns and 'target_binary' in df.columns:
        high_oldpeak = df[df['oldpeak'] > 2]
        if len(high_oldpeak) > 0:
            high_oldpeak_prevalence = high_oldpeak['target_binary'].mean() * 100
            if high_oldpeak_prevalence > 75:
                patterns['risk_profiles'].append(
                    f"ST depression > 2mm strongly associated with disease: {high_oldpeak_prevalence:.1f}% prevalence"
                )

    # Complex patterns - combinations of risk factors
    if all(col in df.columns for col in ['age', 'sex', 'cp', 'thalach', 'target_binary']):
        # High-risk profile: Older men with asymptomatic chest pain and low max heart rate
        high_risk = df[(df['age'] > 55) & (df['sex'] == 1) & (df['cp'] == 4) &
                      (df['thalach'] < df['thalach'].median())]

        if len(high_risk) >= 10:
            high_risk_prevalence = high_risk['target_binary'].mean() * 100
            if high_risk_prevalence > 75:
                patterns['risk_profiles'].append(
                    f"High-risk profile: Men >55 with asymptomatic chest pain and below-median max heart rate ({high_risk_prevalence:.1f}% disease prevalence)"
                )

        # Low-risk profile: Younger women with typical angina and high max heart rate
        low_risk = df[(df['age'] < 45) & (df['sex'] == 0) & (df['cp'] == 1) &
                     (df['thalach'] > df['thalach'].median())]

        if len(low_risk) >= 5:
            low_risk_prevalence = low_risk['target_binary'].mean() * 100
            if low_risk_prevalence < 25:
                patterns['risk_profiles'].append(
                    f"Low-risk profile: Women <45 with typical angina and above-median max heart rate ({low_risk_prevalence:.1f}% disease prevalence)"
                )

    # Areas requiring further investigation based on data quality
    missing_cols = df.columns[df.isnull().mean() > 0.1]
    if len(missing_cols) > 0:
        patterns['investigation_areas'].append(
            f"Investigate impact of missing data in columns: {', '.join(missing_cols)}"
        )

    # Potential non-linear relationships
    continuous_cols = df.select_dtypes(include=['float']).columns
    for col in continuous_cols:
        if 'outlier' not in col and 'missing' not in col and col != 'target_binary':
            # Check skewness
            skew = df[col].skew()
            if abs(skew) > 1:
                patterns['investigation_areas'].append(
                    f"Investigate non-linear relationships for {col} (skewness: {skew:.2f})"
                )

    return patterns


def perform_statistical_analysis(df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive statistical analysis on the heart disease dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset

    Returns
    -------
    Dict
        Dictionary containing all analysis results
    """
    results = {}

    print("Performing comprehensive statistical analysis...")

    # Descriptive statistics
    print("\n1. Calculating descriptive statistics...")
    results['descriptive_statistics'] = calculate_descriptive_statistics(df)

    # Gender analysis
    print("\n2. Performing gender-based analysis...")
    results['gender_analysis'] = segment_analysis_by_gender(df)

    # Age analysis
    print("\n3. Performing age-based analysis...")
    results['age_analysis'] = segment_analysis_by_age(df)

    # Disease status analysis
    print("\n4. Performing disease-based analysis...")
    results['disease_analysis'] = segment_analysis_by_disease(df)

    # Notable patterns
    print("\n5. Identifying notable patterns...")
    results['patterns'] = identify_notable_patterns(df)

    print("\nStatistical analysis complete.")

    return results


if __name__ == "__main__":
    # Test the module if run directly
    print("Heart Disease Dataset Statistical Analysis Module")
    print("This module provides statistical analysis functions for the Heart Disease dataset.")
    print("Import and use in another script or notebook.")