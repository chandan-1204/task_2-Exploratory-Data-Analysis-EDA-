"""
House Price EDA script (ready for Jupyter or VS Code)
- Place your dataset in /mnt/data or in your working folder, or set RAW_PATH explicitly.
- Saves: house_price_eda_report.csv, cleaned_sample.csv, and a few PNG plots.
"""

import os
import fnmatch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -------------------------
# CONFIG - adjust if needed
# -------------------------
RAW_PATH = "C:\\Users\\chand\\Downloads\\house_prices.csv"
SEARCH_DIRS = ['.', '/mnt/data', os.path.expanduser('C:\\Users\\chand\\'), os.path.join(os.path.expanduser('C:\\Users\\chand\\Onedrive\\Documents'),'OneDrive','Documents')]

OUT_DIR = './house_eda_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Helper: find candidate files
# -------------------------
def find_house_file():
    if RAW_PATH:
        return RAW_PATH if os.path.exists(RAW_PATH) else None
    candidates = []
    for d in SEARCH_DIRS:
        try:
            for root, dirs, files in os.walk(d):
                for name in files:
                    if 'house' in name.lower() or 'houseprice' in name.lower() or 'house_price' in name.lower() or 'housing' in name.lower():
                        if name.lower().endswith(('.csv','.xlsx','.xls')):
                            candidates.append(os.path.join(root, name))
        except Exception:
            pass
    return candidates[0] if candidates else None

# -------------------------
# 1. Load dataset
# -------------------------
path = find_house_file()
if path is None:
    raise FileNotFoundError("Could not find a house price dataset automatically. Set RAW_PATH to the CSV/XLSX file path and re-run.")
print("Loading dataset:", path)

# Load CSV or Excel automatically
if path.lower().endswith(('.xls','.xlsx')):
    df = pd.read_excel(path)
else:
    df = pd.read_csv(path)

print("Shape:", df.shape)
display_head = True
if display_head:
    print(df.head(3))

# -------------------------
# 2. Quick overview & basic checks
# -------------------------
print("\n--- Info ---")
print(df.info())

print("\n--- Missing values per column ---")
miss = df.isnull().sum().sort_values(ascending=False)
print(miss[miss>0])

print("\n--- Basic statistics (numeric) ---")
print(df.describe().T)


# Save a snapshot of summary stats to report
report_rows = []
for c in df.columns:
    if pd.api.types.is_numeric_dtype(df[c]):
        report_rows.append({
            'column': c,
            'count': int(df[c].count()),
            'n_missing': int(df[c].isnull().sum()),
            'mean': float(df[c].mean(skipna=True)) if df[c].count()>0 else np.nan,
            'median': float(df[c].median(skipna=True)) if df[c].count()>0 else np.nan,
            'std': float(df[c].std(skipna=True)) if df[c].count()>0 else np.nan,
            'min': float(df[c].min(skipna=True)) if df[c].count()>0 else np.nan,
            'max': float(df[c].max(skipna=True)) if df[c].count()>0 else np.nan,
        })

# -------------------------
# 3. Visualizations (save PNGs)
# -------------------------
# A. Distribution plots for top numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
top_num = num_cols[:6]  # limit to first 6 to avoid huge output

for col in top_num:
    plt.figure(figsize=(6,3))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'{col} distribution')
    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, f'dist_{col}.png')
    plt.savefig(fpath)
    plt.close()

# B. Boxplots to inspect outliers
for col in top_num:
    plt.figure(figsize=(6,2.5))
    sns.boxplot(x=df[col].dropna())
    plt.title(f'{col} boxplot')
    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, f'box_{col}.png')
    plt.savefig(fpath)
    plt.close()

# C. Correlation heatmap (numeric)
if len(num_cols) >= 2:
    corr = df[num_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title('Numeric features correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'correlation_heatmap.png'))
    plt.close()

# D. Pairplot sample (only if not too large)
sample_for_pair = df[num_cols].sample(n=min(200, len(df))) if len(num_cols)>=2 else None
if sample_for_pair is not None and len(num_cols) <= 6:
    sns.pairplot(sample_for_pair)
    plt.savefig(os.path.join(OUT_DIR, 'pairplot_sample.png'))
    plt.close()

print("Saved plots to", OUT_DIR)

# -------------------------
# 4. Missing value handling (suggestions)
# -------------------------
# Default safe choices:
# - numeric columns: median
# - categorical: mode
# - for better fill use KNNImputer on numeric features (code below as option)

df_clean = df.copy()

# Example: fill numeric nulls with median
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
for c in numeric_cols:
    if df_clean[c].isnull().sum() > 0:
        med = df_clean[c].median()
        df_clean[c] = df_clean[c].fillna(med)
        print(f"Filled numeric {c} with median = {med}")

# Example: fill categorical nulls with mode
cat_cols = df_clean.select_dtypes(include=['object','category']).columns.tolist()
for c in cat_cols:
    if df_clean[c].isnull().sum() > 0:
        modev = df_clean[c].mode(dropna=True)[0]
        df_clean[c] = df_clean[c].fillna(modev)
        print(f"Filled categorical {c} with mode = {modev}")

# Option: KNN imputer for numeric (uncomment to use)
# knn = KNNImputer(n_neighbors=5)
# df_clean[numeric_cols] = knn.fit_transform(df_clean[numeric_cols])

print("Missing values after simple imputation:")
print(df_clean.isnull().sum().sort_values(ascending=False).head(20))

# -------------------------
# 5. Feature engineering suggestions (apply sensible ones if columns exist)
# -------------------------
# Common house price features (examples): 'YearBuilt', 'YearRemodAdd', 'LotArea', 'GrLivArea', 'TotalBsmtSF', 'BedroomAbvGr', 'Bathroom'
# Create Age of house if YearBuilt present
if 'YearBuilt' in df_clean.columns:
    this_year = pd.Timestamp.now().year
    df_clean['HouseAge'] = this_year - df_clean['YearBuilt']

# Create total rooms if common columns present
possible_rooms = ['BedroomAbvGr','TotRmsAbvGrd','FullBath','HalfBath']
if all(c in df_clean.columns for c in ['TotRmsAbvGrd']):
    # keep safe: if TotRmsAbvGrd exists, create rooms ratio features
    if 'GrLivArea' in df_clean.columns and 'LotArea' in df_clean.columns:
        df_clean['Price_per_sqft'] = None
        if 'SalePrice' in df_clean.columns:
            # avoid division by zero
            df_clean.loc[df_clean['GrLivArea']>0, 'Price_per_sqft'] = df_clean.loc[df_clean['GrLivArea']>0, 'SalePrice'] / df_clean.loc[df_clean['GrLivArea']>0, 'GrLivArea']

# -------------------------
# 6. Outlier detection & optional capping (IQR)
# -------------------------
def cap_iqr(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k*iqr
    upper = q3 + k*iqr
    return series.clip(lower, upper)

# Example: cap a few numeric cols to reduce extreme outliers (only if they exist)
cap_cols = ['SalePrice','GrLivArea','LotArea']  # typical house-price columns
for c in cap_cols:
    if c in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[c]):
        df_clean[c + '_capped'] = cap_iqr(df_clean[c])
        print(f"Created capped version for {c} -> {c+'_capped'}")

# -------------------------
# 7. Correlation with target (if target present)
# -------------------------
target_candidates = [c for c in df_clean.columns if c.lower() in ('saleprice','price','sale_price','target')]
if target_candidates:
    target = target_candidates[0]
    print("Detected target column:", target)
    # show top correlated features
    corr_with_target = df_clean.select_dtypes(include=[np.number]).corr()[target].abs().sort_values(ascending=False)
    print("\nTop correlations with target:\n", corr_with_target.head(10))
    # Save top list to report
    for col, val in corr_with_target.items():
        report_rows.append({'column': col, 'corr_with_target': float(val)})
else:
    print("No typical target column detected (SalePrice). You can set your target column manually.")

# -------------------------
# 8. Save outputs
# -------------------------
# Save cleaned sample (first N rows) and full cleaned if desired
CLEANED_SAMPLE = os.path.join(OUT_DIR, 'cleaned_sample.csv')
df_clean.head(500).to_csv(CLEANED_SAMPLE, index=False)
print("Saved cleaned sample to:", CLEANED_SAMPLE)

# Save report CSV with stats
report_df = pd.DataFrame(report_rows).drop_duplicates(subset=['column']).reset_index(drop=True)
REPORT_CSV = os.path.join(OUT_DIR, 'house_price_eda_report.csv')
report_df.to_csv(REPORT_CSV, index=False)
print("Saved EDA report to:", REPORT_CSV)

print("EDA complete. Check the output folder for images and CSVs:", OUT_DIR)
