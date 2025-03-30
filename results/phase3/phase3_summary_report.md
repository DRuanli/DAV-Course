# Heart Disease Dataset Analysis - Phase 3 Summary Report
**Generated on:** 2025-03-30 15:55:48

## 1. Probability Distribution Analysis
### Key Findings:
- age: Approximately normal distribution
- chol: Approximately normal distribution
- thalach: Approximately normal distribution
- oldpeak: Moderately non-normal distribution
- trestbps: Approximately normal distribution
- age varies significantly across Disease Status groups (Mann-Whitney U test, p=0.0000)
- age varies significantly across Age Group groups (Kruskal-Wallis H-test, p=0.0000)
- chol varies significantly across Gender groups (Mann-Whitney U test, p=0.0005)
- thalach varies significantly across Disease Status groups (Mann-Whitney U test, p=0.0000)
- thalach varies significantly across Gender groups (Mann-Whitney U test, p=0.0000)
- thalach varies significantly across Age Group groups (Kruskal-Wallis H-test, p=0.0000)
- oldpeak varies significantly across Disease Status groups (Mann-Whitney U test, p=0.0000)
- oldpeak varies significantly across Gender groups (Mann-Whitney U test, p=0.0019)
- oldpeak varies significantly across Age Group groups (Kruskal-Wallis H-test, p=0.0000)
- trestbps varies significantly across Disease Status groups (Mann-Whitney U test, p=0.0017)
- trestbps varies significantly across Age Group groups (Kruskal-Wallis H-test, p=0.0000)

### Normality Test Results:
- **age**: 
  - Shapiro-Wilk Test: Non-normal (p=0.0000)
  - Kolmogorov-Smirnov Test: Non-normal (p=0.0013)
- **chol**: 
  - Shapiro-Wilk Test: Non-normal (p=0.0000)
  - Kolmogorov-Smirnov Test: Non-normal (p=0.0000)
- **thalach**: 
  - Shapiro-Wilk Test: Non-normal (p=0.0000)
  - Kolmogorov-Smirnov Test: Non-normal (p=0.0000)
- **oldpeak**: 
  - Shapiro-Wilk Test: Non-normal (p=0.0000)
  - Kolmogorov-Smirnov Test: Non-normal (p=0.0000)
- **trestbps**: 
  - Shapiro-Wilk Test: Non-normal (p=0.0000)
  - Kolmogorov-Smirnov Test: Non-normal (p=0.0000)

## 2. Hypothesis Testing

## 3. Correlation and Relationship Analysis
### Strongest Correlations:
- num and target_binary: r = 0.783 (p = 0.0000)
- ca and num: r = 0.516 (p = 0.0000)
- thal and target_binary: r = 0.499 (p = 0.0000)
- cp and target_binary: r = 0.495 (p = 0.0000)
- ca and target_binary: r = 0.456 (p = 0.0000)
- thal and num: r = 0.440 (p = 0.0000)
- exang and target_binary: r = 0.434 (p = 0.0000)
- cp and num: r = 0.416 (p = 0.0000)
- cp and exang: r = 0.412 (p = 0.0000)
- oldpeak and num: r = 0.412 (p = 0.0000)

### Key Partial Correlation Insights:
- The correlation between age and cp decreases from 0.18 to 0.00 after controlling for age, sex, suggesting a confounding effect.
- The correlation between age and trestbps decreases from 0.24 to 0.00 after controlling for age, sex, suggesting a confounding effect.
- The correlation between age and fbs decreases from 0.22 to 0.00 after controlling for age, sex, suggesting a confounding effect.
- The correlation between age and restecg decreases from 0.21 to 0.00 after controlling for age, sex, suggesting a confounding effect.
- The correlation between age and thalach decreases from -0.35 to 0.00 after controlling for age, sex, suggesting a confounding effect.

## 4. Conclusion and Clinical Interpretation
The Phase 3 analysis has provided several insights with potential clinical relevance:
- Age distribution appears balanced, suggesting a representative sample across age groups.
- Cholesterol distribution is relatively symmetric, suggesting a well-distributed range of cholesterol levels.
- Maximum heart rate distribution shows expected pattern for a diverse cardiac patient population.
- ST depression distribution is right-skewed, with most patients showing smaller depressions and fewer patients with severe ST depression.
- Distribution of trestbps provides insights into the spread and central tendency of this parameter in the patient population.
- Significant age differences between patients with and without heart disease suggest age is an important risk factor.
- Age-related variations in age suggest the importance of age-stratified approaches to cardiac risk assessment.
- Gender-based differences in chol may indicate the need for gender-specific risk assessment or treatment approaches.
- Significant differences in maximum heart rate between patients with and without heart disease may indicate functional cardiac capacity disparities.
- Gender-based differences in thalach may indicate the need for gender-specific risk assessment or treatment approaches.
- Age-related variations in thalach suggest the importance of age-stratified approaches to cardiac risk assessment.
- Significant differences in ST depression between groups highlight its value as a diagnostic indicator.
- Gender-based differences in oldpeak may indicate the need for gender-specific risk assessment or treatment approaches.
- Age-related variations in oldpeak suggest the importance of age-stratified approaches to cardiac risk assessment.
- Significant differences in trestbps between patients with and without heart disease suggest its potential diagnostic or prognostic value.
- Age-related variations in trestbps suggest the importance of age-stratified approaches to cardiac risk assessment.
- Gender correlates with heart disease (r=0.31), with males showing higher risk in this dataset.
- Chest pain type shows a significant correlation (r=0.49) with heart disease, highlighting the importance of pain characteristics in diagnosis.
- Maximum heart rate achieved shows a negative correlation (r=-0.38) with heart disease, suggesting reduced cardiac functional capacity in disease patients.
- exang shows a strong positive correlation (r=0.43) with heart disease, suggesting its potential importance in risk assessment.
- ST depression (oldpeak) strongly correlates (r=0.37) with heart disease, confirming its value as a diagnostic indicator on ECG.
- ca shows a strong positive correlation (r=0.46) with heart disease, suggesting its potential importance in risk assessment.
- thal shows a strong positive correlation (r=0.50) with heart disease, suggesting its potential importance in risk assessment.
- num shows a strong positive correlation (r=0.78) with heart disease, suggesting its potential importance in risk assessment.
- The correlation between age and cp decreases from 0.18 to 0.00 after controlling for age, sex, suggesting a confounding effect.
- The correlation between age and trestbps decreases from 0.24 to 0.00 after controlling for age, sex, suggesting a confounding effect.
- The correlation between age and fbs decreases from 0.22 to 0.00 after controlling for age, sex, suggesting a confounding effect.