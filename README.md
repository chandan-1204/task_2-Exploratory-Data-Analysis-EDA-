# House Price Dataset – Exploratory Data Analysis (EDA)
## AI & ML Internship – Task 2

This project performs Exploratory Data Analysis (EDA) on a House Price Dataset.  
The goal is to understand the data, discover patterns, identify anomalies, and generate insights using statistical and visual techniques.

-----------------------------------------------------

## Objective
- Understand the dataset using descriptive statistics.
- Explore patterns, trends, and correlations.
- Create visualizations and feature-level insights.

-----------------------------------------------------

## Tools & Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

-----------------------------------------------------

## Project Structure
House_prices_EDA/
│
├── house_price_eda.py
├── README.md
├── house_eda_outputs/
│    ├── cleaned_sample.csv
│    ├── house_price_eda_report.csv
│    ├── dist_<feature>.png
│    ├── box_<feature>.png
│    ├── correlation_heatmap.png
│    └── pairplot_sample.png
└── House-Price-Dataset.csv

-----------------------------------------------------

## Dataset
Use any house price dataset such as:
- Kaggle "House Prices – Advanced Regression Techniques"
- Your own CSV/XLSX file

If needed, update the script:

RAW_PATH = r"C:/path/to/House-Price-Dataset.csv"

-----------------------------------------------------

## Steps Performed in EDA

### 1. Load Dataset
- Auto-detects house price CSV/XLSX
- Reads file using pandas

### 2. Initial Checks
- Dataset shape
- Missing values
- Data types
- Preview of first rows

### 3. Summary Statistics
- Mean, median, std, min, max
- Count and missing value counts
- Saved into house_price_eda_report.csv

### 4. Visualizations
Generated inside `house_eda_outputs/`:
- Histograms (distribution)
- Boxplots (outlier detection)
- Correlation heatmap
- Pairplot (if dataset small enough)

### 5. Missing Value Handling
- Numeric → median
- Categorical → mode
- Optional KNN imputation

### 6. Feature Engineering
- HouseAge = CurrentYear – YearBuilt
- Price_per_sqft = SalePrice / GrLivArea (if available)

### 7. Outlier Handling (IQR)
- Creates capped versions of important numeric features.

### 8. Correlation with Target
- Identifies target column (SalePrice)
- Shows top correlated features
- Saves to report

### 9. Save Outputs
- cleaned_sample.csv  
- house_price_eda_report.csv  
- histogram, boxplot, heatmap, pairplot images  

-----------------------------------------------------

## How to Run

### In VS Code
1. Activate virtual environment:
   venv\Scripts\activate

2. Install dependencies:
   pip install pandas numpy seaborn matplotlib scikit-learn joblib

3. Run:
   python house_price_eda.py


## What You Learn
- Data visualization
- Descriptive statistics
- Correlation analysis
- Outlier detection
- Pattern identification
- Basic feature engineering

-----------------------------------------------------

## Notes
- Works with any housing dataset.
- If using VS Code, replace Jupyter `display()` with `print()`.

-----------------------------------------------------


