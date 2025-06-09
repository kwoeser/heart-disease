# Heart Disease ML Pipeline 

This document describes the preprocessing pipeline used to prepare the Heart Disease dataset for machine learning modeling.

Data from: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

## Pipeline

![Pipeline](../screenshots/pipeline.png)

This Pipeline is responsible for transforming both categorical and numeric features to prepare them for model training. There are 5 numeric features and 1 categorial (sex)

1. **Sex Mapping**
   - `CustomMappingTransformer('Sex', {'M': 0, 'F': 1})`
   - Converts gender into numeric format .

2. **Tukey Outlier Treatment (Outer Fence)**
   - `CustomTukeyTransformer` Applied to: [Age, Cholesterol, MaxHR, Oldpeak, RestingBP]
   - Detects and clips extreme outliers 

3. **Robust Scaling**
   - `CustomRobustTransformer` Applied to: [Age, Cholesterol, MaxHR, Oldpeak, RestingBP]
   - Scales using median and IQR 

4. **KNN Imputation**
   - `CustomKNNTransformer(n_neighbors=4)`
   - Imputes any missing values using k-nearest neighbors.



## Random State

The value used in train_test_split is: 

rs = 95