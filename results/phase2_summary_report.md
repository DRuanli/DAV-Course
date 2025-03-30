# Heart Disease Dataset Analysis - Phase 2 Summary Report
**Generated on:** 2025-03-30 15:26:11

## 1. Dataset Overview
- Original dataset: 920 records, 15 features
- Processed dataset: 920 records, 42 features
- Data sources: cleveland, hungarian, switzerland, va
- Disease prevalence: 55.3%

## 2. Data Preprocessing Summary
- Missing values: 0 values before preprocessing, 1097 values after
- New features created: 27
  - ca_missing, thal_missing, age_outlier, age_original, cp_outlier, cp_original, trestbps_outlier, trestbps_original, chol_outlier, chol_original...

## 3. Key Statistical Findings
### Notable Patterns:
- Notable gender difference in disease prevalence: Males 63.2% vs. Females 25.8%
- Disease prevalence increases monotonically with age
- Notable presence of young patients with heart disease: 58 patients under 45
- High disease prevalence in asymptomatic patients: 79.0% vs. overall 55.3%
- High disease prevalence in patients with low maximum heart rate: 77.4% vs. overall 55.3%

### Top Predictors of Heart Disease:
- cp: Large effect (p=0.0000)
- slope: Large effect (p=nan)
- thal: Large effect (p=0.0000)
- ca: Large effect (p=0.0000)
- exang: Large effect (p=0.0000)

## 4. Visualization Summary
- Created 114 visualizations
- Key visualizations include:
  - Histograms of continuous variables
  - Bar charts of categorical variables
  - Correlation heatmap
  - Boxplots comparing features across disease status

## 5. Areas for Further Investigation
- Strong predictors of heart disease: thal, cp, cp_original, ca, exang, thalach, thalach_original, oldpeak, oldpeak_original, max_hr_pct, sex
- Investigate gender-specific risk factors and diagnostic criteria
- Investigate risk factors for early-onset heart disease
- Investigate diagnostic methods for asymptomatic patients
- Investigate impact of missing data in columns: ca, thal