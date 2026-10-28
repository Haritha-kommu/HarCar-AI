"""
preprocess_data.py
Module 2: Data Preprocessing Script
Project: Used-Car Price Prediction Using Random Forest Regression

This script performs:
1. Loading of raw car data without modifying the source file.
2. Duplicate row detection and removal.
3. Data integrity validation (missing values, data types, physical value bounds).
4. Feature engineering: computes 'Car_Age' using a configurable reference year.
5. Separation of target variable ('Selling_Price') from input features.
6. Generation of cleaned dataset saved to 'data/cleaned_car_data.csv'.
7. Definition of a scikit-learn ColumnTransformer fitted strictly on training data
   (80:20 split, random_state=42) to prevent data leakage.
8. Generation of a detailed preprocessing report saved to 'outputs/preprocessing_report.txt'.
"""

import os
import pandas as pd
import numpy as np

# Configurable Reference Year
# The CarDekho dataset was collected around 2020 (listings up to 2018).
# Setting REFERENCE_YEAR = 2020 provides accurate historical vehicle ages at the time of listing.
DEFAULT_REFERENCE_YEAR = 2020


def load_and_clean_data(raw_data_path: str, reference_year: int = DEFAULT_REFERENCE_YEAR):
    """
    Loads raw data, removes duplicate listings, validates values,
    and engineers the Car_Age feature.
    """
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw dataset not found at: {raw_data_path}")

    # 1. Read raw data without altering the source file
    df_raw = pd.read_csv(raw_data_path)
    original_rows, original_cols = df_raw.shape
    print(f"[INFO] Loaded raw dataset: {original_rows} rows, {original_cols} columns.")

    # 2. Check and remove duplicates
    duplicate_count = df_raw.duplicated().sum()
    print(f"[INFO] Duplicate rows detected: {duplicate_count}")
    df_cleaned = df_raw.drop_duplicates().copy()
    cleaned_rows = df_cleaned.shape[0]
    print(f"[INFO] Dataset shape after duplicate removal: {cleaned_rows} rows.")

    # 3. Validation checks
    # Check nulls
    null_counts = df_cleaned.isnull().sum().sum()
    if null_counts > 0:
        raise ValueError(f"Unexpected missing values found: {null_counts}")
    print("[INFO] Data validation passed: 0 missing values.")

    # Check numerical bounds
    if (df_cleaned['Selling_Price'] <= 0).any():
        raise ValueError("Invalid Selling_Price found (<= 0).")
    if (df_cleaned['Present_Price'] <= 0).any():
        raise ValueError("Invalid Present_Price found (<= 0).")
    if (df_cleaned['Kms_Driven'] < 0).any():
        raise ValueError("Invalid Kms_Driven found (< 0).")
    if (df_cleaned['Year'] > reference_year).any():
        raise ValueError(f"Car manufacturing Year exceeds reference year ({reference_year}).")
    print("[INFO] Numerical bounds validation passed.")

    # 4. Feature Engineering: Car_Age
    # Car_Age = Reference_Year - Year of manufacture
    df_cleaned['Car_Age'] = reference_year - df_cleaned['Year']
    print(f"[INFO] Computed 'Car_Age' using reference year {reference_year}. "
          f"Min age: {df_cleaned['Car_Age'].min()} years, Max age: {df_cleaned['Car_Age'].max()} years.")

    return df_raw, df_cleaned, duplicate_count


def build_preprocessor():
    """
    Creates a scikit-learn ColumnTransformer for categorical and numerical features.
    - Categorical features: One-Hot Encoded (drop='first' to avoid multicollinearity)
    - Numerical features: Passed through without scaling (Random Forest is invariant to monotonic scaling)
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
    numerical_features = ['Present_Price', 'Kms_Driven', 'Owner', 'Car_Age']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor, numerical_features, categorical_features


def split_and_validate(df_cleaned: pd.DataFrame, preprocessor, test_size: float = 0.2, random_state: int = 42):
    """
    Splits features and target into train (80%) and test (20%) sets.
    Fits preprocessor strictly on training data to prevent data leakage.
    """
    from sklearn.model_selection import train_test_split

    # Input features (excluding target 'Selling_Price' and raw 'Car_Name', 'Year')
    # Note: 'Car_Name' has high cardinality (98 unique names) and is excluded from
    # tabular regression to prevent extreme sparsity and overfitting.
    feature_cols = ['Present_Price', 'Kms_Driven', 'Owner', 'Car_Age', 'Fuel_Type', 'Seller_Type', 'Transmission']
    target_col = 'Selling_Price'

    X = df_cleaned[feature_cols].copy()
    y = df_cleaned[target_col].copy()

    # 80:20 Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\n[INFO] Train-Test Split (test_size={test_size}, random_state={random_state}):")
    print(f"   - Training set size: {X_train.shape[0]} samples")
    print(f"   - Testing set size:  {X_test.shape[0]} samples")

    # Fit preprocessor strictly on X_train to prevent data leakage
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Get feature names after one-hot encoding
    encoded_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(['Fuel_Type', 'Seller_Type', 'Transmission']).tolist()
    numerical_features = ['Present_Price', 'Kms_Driven', 'Owner', 'Car_Age']
    all_transformed_feature_names = numerical_features + encoded_cat_features

    print(f"[INFO] Transformed feature count: {len(all_transformed_feature_names)}")
    print(f"[INFO] Encoded feature names: {all_transformed_feature_names}")

    split_info = {
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "input_features": feature_cols,
        "transformed_features": all_transformed_feature_names,
        "test_size": test_size,
        "random_state": random_state
    }

    return X_train, X_test, y_train, y_test, split_info


def save_cleaned_data_and_report(df_raw: pd.DataFrame, df_cleaned: pd.DataFrame,
                                 duplicate_count: int, split_info: dict,
                                 output_csv_path: str, report_path: str,
                                 reference_year: int = DEFAULT_REFERENCE_YEAR):
    """
    Saves the cleaned dataset and writes a comprehensive preprocessing report.
    """
    # 1. Save cleaned dataset
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_cleaned.to_csv(output_csv_path, index=False)
    print(f"\n[SUCCESS] Cleaned dataset saved to: {output_csv_path}")

    # 2. Generate Preprocessing Report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report_content = f"""================================================================================
MODULE 2: DATA PREPROCESSING REPORT
Project: Used-Car Price Prediction Using Random Forest Regression
================================================================================

1. DATASET VOLUME & DEDUPLICATION:
--------------------------------------------------------------------------------
- Original Raw Records (data/car_data.csv) : {df_raw.shape[0]} rows, {df_raw.shape[1]} columns
- Duplicate Rows Identified & Removed     : {duplicate_count} rows
- Final Cleaned Records                   : {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns

2. INTEGRITY CHECKS:
--------------------------------------------------------------------------------
- Missing / Null Values                   : 0 (No imputation needed)
- Negative / Zero Value Inconsistencies   : None detected
- Target Variable Range (Selling_Price)   : ₹{df_cleaned['Selling_Price'].min():.2f} Lakhs to ₹{df_cleaned['Selling_Price'].max():.2f} Lakhs
- Ex-Showroom Price Range (Present_Price) : ₹{df_cleaned['Present_Price'].min():.2f} Lakhs to ₹{df_cleaned['Present_Price'].max():.2f} Lakhs
- Odometer Range (Kms_Driven)             : {df_cleaned['Kms_Driven'].min():,} km to {df_cleaned['Kms_Driven'].max():,} km

3. FEATURE ENGINEERING (CAR AGE):
--------------------------------------------------------------------------------
- Reference Year Used                     : {reference_year}
- Justification                           : Historical CarDekho dataset was compiled around 2020 
                                            (vehicle years span 2003-2018). Using 2020 reflects the 
                                            actual vehicle age at the time of market listing.
- Formula                                 : Car_Age = {reference_year} - Year
- Vehicle Age Range in Cleaned Data       : {df_cleaned['Car_Age'].min()} to {df_cleaned['Car_Age'].max()} years
- Raw 'Year' Column Handling              : Retained in cleaned CSV for auditability, but replaced 
                                            by 'Car_Age' in the ML feature set.

4. FEATURE SELECTION & RATIONALE:
--------------------------------------------------------------------------------
- Target Variable (y)                     : Selling_Price (Preserved strictly as output; never in X)
- Input Features Selected (X)             :
    1. Present_Price (Numerical, Lakhs INR)
    2. Kms_Driven    (Numerical, Kilometers)
    3. Owner         (Numerical, Previous Owners Count: 0, 1, 3)
    4. Car_Age       (Numerical, Derived Years)
    5. Fuel_Type     (Categorical: Petrol, Diesel, CNG)
    6. Seller_Type   (Categorical: Dealer, Individual)
    7. Transmission  (Categorical: Manual, Automatic)
- Excluded Features:
    * Car_Name       : High cardinality (98 distinct models in 299 rows). Excluded from 
                       regression input to prevent extreme sparsity and model overfitting.
    * Year           : Excluded to eliminate strict collinearity with 'Car_Age'.

5. TRAIN / TEST SPLIT (NO DATA LEAKAGE):
--------------------------------------------------------------------------------
- Split Ratio                             : 80% Training ({split_info['train_samples']} samples), 20% Testing ({split_info['test_samples']} samples)
- Random Seed                             : {split_info['random_state']}
- Preprocessing Pipeline Strategy         :
    * ColumnTransformer with OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    * Preprocessor is FITTED ONLY ON TRAINING SET (X_train) and applied to test set (X_test).
    * No whole-dataset transformation occurred prior to splitting.
- Final Transformed Features ({len(split_info['transformed_features'])} total) :
    {split_info['transformed_features']}

6. SAVED ARTIFACTS:
--------------------------------------------------------------------------------
- Cleaned CSV                             : data/cleaned_car_data.csv
- Preprocessing Report                    : outputs/preprocessing_report.txt
================================================================================
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[SUCCESS] Preprocessing report saved to: {report_path}")


def main():
    print("=" * 70)
    print("USED-CAR PRICE PREDICTION - DATA PREPROCESSING (MODULE 2)")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, "data", "car_data.csv")
    cleaned_path = os.path.join(base_dir, "data", "cleaned_car_data.csv")
    report_path = os.path.join(base_dir, "outputs", "preprocessing_report.txt")

    # Step 1: Load, Deduplicate, Validate & Engineer Car_Age
    df_raw, df_cleaned, dup_count = load_and_clean_data(raw_path, reference_year=DEFAULT_REFERENCE_YEAR)

    # Step 2: Build Preprocessor (ColumnTransformer)
    preprocessor, num_cols, cat_cols = build_preprocessor()

    # Step 3: Train-Test Split & Fit Preprocessor on Training Set Only
    X_train, X_test, y_train, y_test, split_info = split_and_validate(
        df_cleaned, preprocessor, test_size=0.2, random_state=42
    )

    # Step 4: Save cleaned dataset and report
    save_cleaned_data_and_report(
        df_raw, df_cleaned, dup_count, split_info,
        cleaned_path, report_path, reference_year=DEFAULT_REFERENCE_YEAR
    )

    print("\n" + "=" * 70)
    print("[COMPLETED] Module 2 Data Preprocessing executed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
