### Chapter Summaries

### Chapter 1: DataFrames, Feature Engineering, and Exploratory Tools

This chapter introduced foundational skills in data manipulation using **pandas**. The focus was on understanding how to:

- Load and explore tabular data with `.head()`, `.info()`, and `.describe()`
- Access and modify data using `.loc[]`, `.iloc[]`, boolean masking, and method chaining
- Perform **column-wise transformations** using `.map()`, `.apply()`, and conditional logic
- Clean and engineer features manually, including:
  - Mapping categorical values to numerical codes
  - Creating new features from existing ones
  - Handling missing or inconsistent data

The exercises used the **Titanic dataset** to explore survival patterns and prepare features for downstream machine learning.

---

### Chapter 2: Custom Transformers and Data Pipelines

This chapter focused on building reusable, clean preprocessing logic using **scikit-learn pipelines**. Key topics included:

- Creating **custom transformers** by extending `BaseEstimator` and `TransformerMixin`
- Writing transformers for:
  - `CustomMappingTransformer`: mapping text categories to numeric codes
  - `CustomOHETransformer`: one-hot encoding a single column using `pd.get_dummies`
  - `CustomDropColumnsTransformer`: dropping or keeping specified columns with error/warning handling
- Assembling a full **Pipeline** to automate data preprocessing for a customer ratings dataset
- Steps included: dropping IDs, mapping gender and experience level, and one-hot encoding OS and ISP fields

By the end of the chapter, we constructed a robust and modular pipeline that can be reused and extended for model training and evaluation.

---

### Chapter 3: GitHub Libraries and Correlation-Based Feature Selection

This chapter focused on organizing reusable code and enhancing feature engineering through correlation analysis.

- Introduced the use of a **GitHub-hosted `library.py`** file to store and reuse custom transformers across notebooks
- Practiced importing library code using `wget` and `%run` in Jupyter/Colab
- Explored **Pearson correlation** to identify multicollinearity between features
- Applied **NumPy masking** (`np.triu`, `np.fill_diagonal`) to isolate the upper triangle of a correlation matrix
- Built `CustomPearsonTransformer`:
  - Calculates correlation in `fit()` and identifies highly correlated columns
  - Drops correlated columns in `transform()` with an error check if `fit()` hasn't been called

This chapter emphasized **modular design**, **code reuse**, and **data quality improvements** in preprocessing workflows.

---

### Chapter 4: Outlier Handling and Clipping with Custom Transformers

This chapter focused on managing outliers using statistical rules and expanding the transformer library with new tools.

- Introduced the **3-sigma rule** for detecting extreme values
- Created `CustomSigma3Transformer` to clip a column within ±3 standard deviations of the mean
- Created `CustomTukeyTransformer` to clip values using Tukey’s **inner** or **outer fences** based on IQR
- Both transformers were added to `library.py`, followed scikit-learn conventions, and included clear assertion checks
- Compared `.query()` and `.drop()` for row filtering and noted their handling of NaNs

By the end, we had reusable tools for clipping outliers and a deeper understanding of how different statistical thresholds affect the data.

---
