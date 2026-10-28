"""
inspect_data.py
Module 1: Dataset Inspection and Exploration Script
Project: Used-Car Price Prediction Using Random Forest Regression
"""

import os
import pandas as pd

def inspect_dataset(file_path: str):
    """
    Loads and inspects the used car dataset, printing comprehensive
    diagnostics and structural information for beginner understanding.
    """
    print("=" * 70)
    print("USED-CAR PRICE PREDICTION - DATASET INSPECTION (MODULE 1)")
    print("=" * 70)

    # 1. Check if the dataset file exists
    if not os.path.exists(file_path):
        print(f"[ERROR] Dataset file not found at: {file_path}")
        return

    # 2. Load the dataset
    print(f"\n[INFO] Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print("[SUCCESS] Dataset loaded successfully!")

    # 3. Shape of the dataset
    print("\n" + "-" * 40)
    print(f"1. DATASET DIMENSIONS (SHAPE):")
    print(f"   - Total Rows (Car listings): {df.shape[0]}")
    print(f"   - Total Columns (Attributes): {df.shape[1]}")

    # 4. First 5 Rows
    print("\n" + "-" * 40)
    print("2. FIRST 5 ROWS (HEAD):")
    print(df.head().to_string())

    # 5. Column Names and Data Types
    print("\n" + "-" * 40)
    print("3. COLUMNS & DATA TYPES:")
    info_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Non-Null Count": df.notnull().sum(),
        "Unique Values": df.nunique()
    })
    print(info_df.to_string(index=False))

    # 6. Missing Values Check
    print("\n" + "-" * 40)
    print("4. MISSING / NULL VALUES COUNT:")
    missing = df.isnull().sum()
    print(missing.to_string())
    total_missing = missing.sum()
    if total_missing == 0:
        print("   -> No missing values detected in the dataset!")
    else:
        print(f"   -> Warning: {total_missing} missing values detected.")

    # 7. Duplicate Rows Check
    print("\n" + "-" * 40)
    print("5. DUPLICATE ROWS CHECK:")
    duplicates = df.duplicated().sum()
    print(f"   - Number of duplicate rows: {duplicates}")
    if duplicates > 0:
        print(f"   -> Notice: {duplicates} duplicate row(s) found. We will handle these during preprocessing.")

    # 8. Summary Statistics (Numerical Features)
    print("\n" + "-" * 40)
    print("6. SUMMARY STATISTICS (NUMERICAL COLUMNS):")
    print(df.describe().round(2).to_string())

    # 9. Summary Statistics (Categorical Features)
    print("\n" + "-" * 40)
    print("7. CATEGORICAL COLUMNS SUMMARY & UNIQUE VALUES:")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        unique_vals = df[col].unique()
        if len(unique_vals) <= 10:
            print(f"   - {col} ({len(unique_vals)} unique): {list(unique_vals)}")
        else:
            print(f"   - {col} ({len(unique_vals)} unique): e.g. {list(unique_vals[:5])}...")

    # 10. Target Column & Feature Identification
    print("\n" + "=" * 70)
    print("8. TARGET COLUMN & FEATURE IDENTIFICATION:")
    print("   * TARGET (y): 'Selling_Price' (continuous numerical value in Lakhs INR)")
    print("   * INPUT FEATURES (X):")
    print("     - Numerical: 'Year' (can be converted to Car Age), 'Present_Price' (ex-showroom in Lakhs INR), 'Kms_Driven', 'Owner'")
    print("     - Categorical: 'Fuel_Type' (Petrol/Diesel/CNG), 'Seller_Type' (Dealer/Individual), 'Transmission' (Manual/Automatic)")
    print("     - Identifier/High Cardinality: 'Car_Name' (name/model of the car)")

    # 11. Service History Check
    print("\n" + "=" * 70)
    print("9. SERVICE-HISTORY INFORMATION CHECK:")
    service_cols = [c for c in df.columns if any(k in c.lower() for k in ['service', 'history', 'maintenance', 'condition'])]
    if service_cols:
        print(f"   - Service history columns found: {service_cols}")
    else:
        print("   - Result: Service-history information is NOT present in this dataset.")
        print("   - Note: The dataset contains vehicle age, odometer reading (Kms_Driven), and owner count as proxy wear indicators.")
    print("=" * 70)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "data", "car_data.csv")
    inspect_dataset(data_file)
