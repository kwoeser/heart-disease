# Heart Disease ML Pipeline Description

This document describes the preprocessing pipeline used to prepare the selected features from the Heart Disease dataset for machine learning modeling.

Data from: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

## Final Feature Set

After analysis, a final set of 7 features was selected, comprising 4 numeric and 3 categorical features.

*   **Numeric Features:** `Cholesterol`, `MaxHR`, `Age`, `RestingBP`
*   **Categorical Features:** `Sex`, `ST_Slope`, `RestingECG`

## Pipeline

![Pipeline](../screenshots/pipeline.png)

The pipeline consists of 12 steps to process the data:

1.  **Tukey Outlier Treatment (Numeric Features)**
    *   `CustomTukeyTransformer('Cholesterol', 'outer')`
    *   `CustomTukeyTransformer('MaxHR', 'outer')`
    *   `CustomTukeyTransformer('Age', 'outer')`
    *   `CustomTukeyTransformer('RestingBP', 'outer')`
    *   These steps identify and cap extreme outliers in each of the four numeric columns.

2.  **Robust Scaling (Numeric Features)**
    *   `CustomRobustTransformer('Cholesterol')`
    *   `CustomRobustTransformer('MaxHR')`
    *   `CustomRobustTransformer('Age')`
    *   `CustomRobustTransformer('RestingBP')`
    *   These steps scale each numeric feature making it robust.

3.  **One-Hot Encoding (Categorical Features)**
    *   `CustomOHETransformer('Sex')`
    *   `CustomOHETransformer('ST_Slope')`
    *   `CustomOHETransformer('RestingECG')`
    *   These steps convert the three categorical columns into a numerical format.

4.  **KNN Imputation**
    *   `CustomKNNTransformer(n_neighbors=5)`
    *   Imputes any missing values in the dataset using the k-nearest neighbors approach.

## Random State

The value used in train_test_split is: 

rs = 102