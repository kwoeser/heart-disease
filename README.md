### 📘 Chapter Summaries

### ✅ Chapter 1: DataFrames, Feature Engineering, and Exploratory Tools

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

### ✅ Chapter 2: Custom Transformers and Data Pipelines

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

