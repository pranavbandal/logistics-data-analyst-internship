"""
clean_data.py
Data cleaning & preprocessing pipeline for the logistics shipment dataset.
Steps: standardize text, remove duplicates, handle missing values,
detect/treat outliers (IQR method), normalize numeric features.
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 120)

df = pd.read_csv("raw_logistics_data.csv")
report_lines = []

def log(msg):
    print(msg)
    report_lines.append(msg)

log("=" * 60)
log("STEP 0: RAW DATA OVERVIEW")
log("=" * 60)
log(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
log(f"Missing values:\n{df.isna().sum().to_string()}")
n_dupes = df.drop(columns=['Shipment_ID']).duplicated().sum()
log(f"Duplicate rows (ignoring ID): {n_dupes}")

# ---------------------------------------------------------------
# STEP 1: Standardize text fields (case, whitespace)
# ---------------------------------------------------------------
log("\n" + "=" * 60)
log("STEP 1: STANDARDIZE TEXT FIELDS")
log("=" * 60)
text_cols = ["Origin_City", "Destination_City", "Carrier", "Transport_Mode", "On_Time"]
before_unique = {c: df[c].nunique(dropna=True) for c in text_cols}

for c in text_cols:
    df[c] = df[c].astype(str).str.strip().str.title()
    df.loc[df[c] == "Nan", c] = np.nan

after_unique = {c: df[c].nunique(dropna=True) for c in text_cols}
for c in text_cols:
    log(f"{c}: {before_unique[c]} unique values -> {after_unique[c]} unique values after standardization")

# ---------------------------------------------------------------
# STEP 2: Remove duplicate rows
# ---------------------------------------------------------------
log("\n" + "=" * 60)
log("STEP 2: REMOVE DUPLICATES")
log("=" * 60)
rows_before = len(df)
df = df.drop_duplicates(subset=[c for c in df.columns if c != "Shipment_ID"], keep="first")
rows_after = len(df)
log(f"Rows before: {rows_before} | Rows after: {rows_after} | Removed: {rows_before - rows_after}")

# ---------------------------------------------------------------
# STEP 3: Handle missing values
# ---------------------------------------------------------------
log("\n" + "=" * 60)
log("STEP 3: HANDLE MISSING VALUES")
log("=" * 60)
log(f"Missing values before imputation:\n{df.isna().sum().to_string()}")

# Numeric columns -> median imputation (robust to outliers/skew)
num_cols = ["Weight_KG", "Delivery_Time_Days", "Shipping_Cost_INR"]
for c in num_cols:
    median_val = df[c].median()
    n_missing = df[c].isna().sum()
    df[c] = df[c].fillna(median_val)
    log(f"{c}: filled {n_missing} missing values with median = {median_val:.2f}")

# Categorical columns -> mode imputation
cat_cols = ["Carrier", "Destination_City"]
for c in cat_cols:
    mode_val = df[c].mode()[0]
    n_missing = df[c].isna().sum()
    df[c] = df[c].fillna(mode_val)
    log(f"{c}: filled {n_missing} missing values with mode = '{mode_val}'")

log(f"\nMissing values after imputation:\n{df.isna().sum().to_string()}")

# ---------------------------------------------------------------
# STEP 4: Outlier detection & treatment (IQR method)
# ---------------------------------------------------------------
log("\n" + "=" * 60)
log("STEP 4: OUTLIER DETECTION (IQR METHOD)")
log("=" * 60)

outlier_cols = ["Distance_KM", "Delivery_Time_Days", "Shipping_Cost_INR", "Weight_KG"]
outlier_summary = {}

for c in outlier_cols:
    Q1 = df[c].quantile(0.25)
    Q3 = df[c].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[c] < lower) | (df[c] > upper)).sum()
    outlier_summary[c] = (lower, upper, n_outliers)
    log(f"{c}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}, bounds=[{lower:.2f}, {upper:.2f}], outliers found={n_outliers}")

# Cap (winsorize) outliers rather than deleting rows, to preserve sample size
log("\nTreatment: capping outliers at the IQR bounds (winsorization)")
for c in outlier_cols:
    lower, upper, _ = outlier_summary[c]
    df[c] = df[c].clip(lower=lower, upper=upper)

# Re-check after treatment
for c in outlier_cols:
    Q1 = df[c].quantile(0.25)
    Q3 = df[c].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[c] < lower) | (df[c] > upper)).sum()
    log(f"{c}: outliers remaining after capping = {n_outliers}")

# ---------------------------------------------------------------
# STEP 5: Normalization (Min-Max scaling) for modeling-ready columns
# ---------------------------------------------------------------
log("\n" + "=" * 60)
log("STEP 5: NORMALIZATION (MIN-MAX SCALING)")
log("=" * 60)

norm_cols = ["Distance_KM", "Weight_KG", "Delivery_Time_Days", "Shipping_Cost_INR"]
for c in norm_cols:
    min_v, max_v = df[c].min(), df[c].max()
    df[c + "_norm"] = (df[c] - min_v) / (max_v - min_v)
    log(f"{c}: min={min_v:.2f}, max={max_v:.2f} -> new column '{c}_norm' scaled to [0, 1]")

# ---------------------------------------------------------------
# STEP 6: Save cleaned dataset + summary stats
# ---------------------------------------------------------------
df.to_csv("cleaned_logistics_data.csv", index=False)

log("\n" + "=" * 60)
log("STEP 6: FINAL SUMMARY")
log("=" * 60)
log(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
log(f"Remaining missing values: {df.isna().sum().sum()}")
log("\nDescriptive statistics (post-cleaning):")
log(df[["Distance_KM", "Weight_KG", "Delivery_Time_Days", "Shipping_Cost_INR"]].describe().round(2).to_string())

with open("cleaning_log.txt", "w") as f:
    f.write("\n".join(report_lines))

print("\nSaved: cleaned_logistics_data.csv, cleaning_log.txt")
