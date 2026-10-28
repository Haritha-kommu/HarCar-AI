"""
eda.py
Module 3: Exploratory Data Analysis (EDA) Script
Project: Used-Car Price Prediction Using Random Forest Regression

This script:
1. Loads the cleaned dataset (data/cleaned_car_data.csv).
2. Computes summary statistics, distribution metrics, and correlation coefficients.
3. Generates high-quality visualizations saved to the outputs/ folder:
   - 01_selling_price_distribution.png: Target distribution histogram with KDE
   - 02_selling_price_boxplot.png: Target boxplot highlighting upper outliers
   - 03_car_age_vs_selling_price.png: Scatter plot with trend line for age vs. price
   - 04_kms_driven_vs_selling_price.png: Odometer distance vs. selling price
   - 05_present_price_vs_selling_price.png: Original showroom price vs. selling price
   - 06_categorical_vs_selling_price.png: Comparison by Fuel Type, Seller Type, Transmission
   - 07_correlation_heatmap.png: Correlation heatmap across numerical attributes
4. Produces outputs/eda_report.txt detailing exact statistics and analytical insights.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output encoding for Windows terminal compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Set publication-quality style for plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads the cleaned dataset and validates presence."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {file_path}")
    df = pd.read_csv(file_path)
    print(f"[INFO] Loaded cleaned dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def generate_plots(df: pd.DataFrame, output_dir: str):
    """Generates and saves all required EDA plots."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Generating plots and saving to: {output_dir}")

    # 1. Selling Price Distribution Histogram with KDE
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Selling_Price'], kde=True, color='#2563EB', bins=25, ax=ax, edgecolor='black', alpha=0.7)
    mean_val = df['Selling_Price'].mean()
    median_val = df['Selling_Price'].median()
    ax.axvline(mean_val, color='#DC2626', linestyle='--', linewidth=2, label=f'Mean: Rs. {mean_val:.2f}L')
    ax.axvline(median_val, color='#16A34A', linestyle='-', linewidth=2, label=f'Median: Rs. {median_val:.2f}L')
    ax.set_title('Distribution of Used Car Selling Price (Target Variable)', fontweight='bold', pad=12)
    ax.set_xlabel('Selling Price (in Lakhs INR)')
    ax.set_ylabel('Number of Cars (Frequency)')
    ax.legend(frameon=True)
    plt.tight_layout()
    p1 = os.path.join(output_dir, '01_selling_price_distribution.png')
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    print("   [+] Saved 01_selling_price_distribution.png")

    # 2. Boxplot of Selling Price to inspect outliers
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df['Selling_Price'], color='#93C5FD', flierprops={'marker': 'o', 'markersize': 6, 'markerfacecolor': '#DC2626'}, ax=ax)
    ax.set_title('Boxplot of Selling Price (Outlier Inspection)', fontweight='bold', pad=12)
    ax.set_xlabel('Selling Price (in Lakhs INR)')
    plt.tight_layout()
    p2 = os.path.join(output_dir, '02_selling_price_boxplot.png')
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    print("   [+] Saved 02_selling_price_boxplot.png")

    # 3. Car Age vs Selling Price (Scatter + Trend line)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=df, x='Car_Age', y='Selling_Price', ax=ax,
                scatter_kws={'alpha': 0.6, 'color': '#2563EB'},
                line_kws={'color': '#DC2626', 'linewidth': 2})
    ax.set_title('Car Age vs. Selling Price (Depreciation Pattern)', fontweight='bold', pad=12)
    ax.set_xlabel('Car Age (Years, Reference Year: 2020)')
    ax.set_ylabel('Selling Price (in Lakhs INR)')
    plt.tight_layout()
    p3 = os.path.join(output_dir, '03_car_age_vs_selling_price.png')
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    print("   [+] Saved 03_car_age_vs_selling_price.png")

    # 4. Kilometers Driven vs Selling Price
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='Kms_Driven', y='Selling_Price', hue='Fuel_Type',
                    palette='tab10', alpha=0.75, s=60, ax=ax)
    ax.set_title('Kilometers Driven vs. Selling Price', fontweight='bold', pad=12)
    ax.set_xlabel('Kilometers Driven (Odometer Reading)')
    ax.set_ylabel('Selling Price (in Lakhs INR)')
    ax.legend(title='Fuel Type', frameon=True)
    plt.tight_layout()
    p4 = os.path.join(output_dir, '04_kms_driven_vs_selling_price.png')
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    print("   [+] Saved 04_kms_driven_vs_selling_price.png")

    # 5. Present Price (Ex-Showroom) vs Selling Price
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=df, x='Present_Price', y='Selling_Price', ax=ax,
                scatter_kws={'alpha': 0.6, 'color': '#7C3AED'},
                line_kws={'color': '#F59E0B', 'linewidth': 2})
    ax.set_title('Original Ex-Showroom Price (Present_Price) vs. Selling Price', fontweight='bold', pad=12)
    ax.set_xlabel('Present Price when New (in Lakhs INR)')
    ax.set_ylabel('Resale Selling Price (in Lakhs INR)')
    plt.tight_layout()
    p5 = os.path.join(output_dir, '05_present_price_vs_selling_price.png')
    fig.savefig(p5, dpi=300)
    plt.close(fig)
    print("   [+] Saved 05_present_price_vs_selling_price.png")

    # 6. Categorical Comparisons: Fuel Type, Seller Type, Transmission
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.boxplot(data=df, x='Fuel_Type', y='Selling_Price', hue='Fuel_Type', legend=False, palette='Blues_r', ax=axes[0])
    axes[0].set_title('Selling Price by Fuel Type', fontweight='bold')
    axes[0].set_xlabel('Fuel Type')
    axes[0].set_ylabel('Selling Price (in Lakhs INR)')

    sns.boxplot(data=df, x='Seller_Type', y='Selling_Price', hue='Seller_Type', legend=False, palette='Greens_r', ax=axes[1])
    axes[1].set_title('Selling Price by Seller Type', fontweight='bold')
    axes[1].set_xlabel('Seller Type')
    axes[1].set_ylabel('')

    sns.boxplot(data=df, x='Transmission', y='Selling_Price', hue='Transmission', legend=False, palette='Oranges_r', ax=axes[2])
    axes[2].set_title('Selling Price by Transmission', fontweight='bold')
    axes[2].set_xlabel('Transmission')
    axes[2].set_ylabel('')

    plt.suptitle('Categorical Feature Comparisons against Selling Price', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    p6 = os.path.join(output_dir, '06_categorical_vs_selling_price.png')
    fig.savefig(p6, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("   [+] Saved 06_categorical_vs_selling_price.png")

    # 7. Correlation Heatmap for Numerical Features
    fig, ax = plt.subplots(figsize=(7, 6))
    num_cols = ['Selling_Price', 'Present_Price', 'Kms_Driven', 'Car_Age', 'Owner']
    corr_matrix = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
                cbar_kws={'label': 'Pearson Correlation Coefficient'}, ax=ax)
    ax.set_title('Correlation Heatmap (Numerical Features)', fontweight='bold', pad=12)
    plt.tight_layout()
    p7 = os.path.join(output_dir, '07_correlation_heatmap.png')
    fig.savefig(p7, dpi=300)
    plt.close(fig)
    print("   [+] Saved 07_correlation_heatmap.png")


def generate_eda_report(df: pd.DataFrame, report_path: str):
    """Computes exact statistics and writes outputs/eda_report.txt."""
    # Target statistics
    y = df['Selling_Price']
    count = len(y)
    mean = y.mean()
    median = y.median()
    std = y.std()
    min_val = y.min()
    max_val = y.max()
    q25 = y.quantile(0.25)
    q75 = y.quantile(0.75)
    iqr = q75 - q25
    skewness = y.skew()
    upper_threshold = q75 + 1.5 * iqr
    outlier_count = (y > upper_threshold).sum()

    # Numerical correlations with target
    num_cols = ['Selling_Price', 'Present_Price', 'Kms_Driven', 'Car_Age', 'Owner']
    corr = df[num_cols].corr()['Selling_Price'].drop('Selling_Price')

    # Categorical statistics
    fuel_means = df.groupby('Fuel_Type')['Selling_Price'].agg(['count', 'mean', 'median'])
    seller_means = df.groupby('Seller_Type')['Selling_Price'].agg(['count', 'mean', 'median'])
    trans_means = df.groupby('Transmission')['Selling_Price'].agg(['count', 'mean', 'median'])

    report_content = f"""================================================================================
MODULE 3: EXPLORATORY DATA ANALYSIS (EDA) REPORT
Project: Used-Car Price Prediction Using Random Forest Regression
================================================================================

1. DATASET OVERVIEW:
--------------------------------------------------------------------------------
- Cleaned Records Analyzed : {count} car listings
- Total Columns            : {df.shape[1]} columns
- Target Variable          : Selling_Price (Unit: Lakhs INR)

2. TARGET VARIABLE STATISTICS (Selling_Price):
--------------------------------------------------------------------------------
- Count                   : {count}
- Mean                    : Rs. {mean:.2f} Lakhs (Rs. {mean*100000:,.0f})
- Median (50th percentile): Rs. {median:.2f} Lakhs (Rs. {median*100000:,.0f})
- Standard Deviation      : Rs. {std:.2f} Lakhs
- Minimum                 : Rs. {min_val:.2f} Lakhs (Rs. {min_val*100000:,.0f})
- 25th Percentile (Q1)    : Rs. {q25:.2f} Lakhs
- 75th Percentile (Q3)    : Rs. {q75:.2f} Lakhs
- Maximum                 : Rs. {max_val:.2f} Lakhs (Rs. {max_val*100000:,.0f})
- Interquartile Range(IQR): Rs. {iqr:.2f} Lakhs
- Skewness                : {skewness:.2f} (Positively / Right-skewed distribution)
- Outlier Threshold (Q3+1.5*IQR) : Rs. {upper_threshold:.2f} Lakhs
- High-Value Outlier Count: {outlier_count} vehicles listed above Rs. {upper_threshold:.2f} Lakhs
  (Note: Kept in dataset as genuine luxury/premium vehicles; Random Forest is naturally
   robust to target and feature outliers without requiring artificial deletion).

3. CORRELATION WITH TARGET (Selling_Price):
--------------------------------------------------------------------------------
{corr.to_string()}

Key Observations:
* Present_Price (+{corr['Present_Price']:.2f}): Strong positive linear association. Cars with 
  higher original showroom cost consistently command higher secondary market resale value.
* Car_Age ({corr['Car_Age']:.2f}): Moderate negative association. Older vehicles suffer from 
  cumulative physical and market depreciation.
* Kms_Driven ({corr['Kms_Driven']:.2f}): Weak positive/flat linear correlation. Higher odometer 
  readings appear in larger/diesel cars that happen to be costlier models.
* Owner ({corr['Owner']:.2f}): Weak negative correlation. Multiple ownership marginally dampens price.

4. CATEGORICAL SUBGROUP ANALYSIS:
--------------------------------------------------------------------------------
A. By Fuel Type:
{fuel_means.round(2).to_string()}
- Insight: Diesel cars exhibit a noticeably higher mean price (Rs. {fuel_means.loc['Diesel', 'mean']:.2f}L) 
  compared to Petrol cars (Rs. {fuel_means.loc['Petrol', 'mean']:.2f}L), primarily because larger SUVs 
  and sedans in the dataset are predominantly diesel-powered.

B. By Seller Type:
{seller_means.round(2).to_string()}
- Insight: Dealer listings have a substantially higher mean selling price (Rs. {seller_means.loc['Dealer', 'mean']:.2f}L) 
  than Individual seller listings (Rs. {seller_means.loc['Individual', 'mean']:.2f}L). Many low-cost two-wheelers 
  or entry-level hatchbacks in the dataset were sold directly by individual owners.

C. By Transmission:
{trans_means.round(2).to_string()}
- Insight: Automatic cars command a higher average resale price (Rs. {trans_means.loc['Automatic', 'mean']:.2f}L) 
  than Manual vehicles (Rs. {trans_means.loc['Manual', 'mean']:.2f}L), reflecting higher baseline showroom costs 
  and consumer willingness to pay for premium convenience.

5. METHODOLOGICAL NOTE ON CAUSATION VS. CORRELATION:
--------------------------------------------------------------------------------
- IMPORTANT: The statistical correlations and scatter patterns identified above 
  represent empirical associations observed within this sample dataset. 
  Correlation does NOT prove causation.
  For example, high kilometers driven does not cause a car to become more expensive; 
  rather, commercial and luxury diesel vehicles happen to both be driven more kilometers 
  and maintain a higher absolute baseline value.
- The non-linear splits in Random Forest Regression will naturally capture these 
  multi-dimensional interactions without assuming simple linear causality.

6. SAVED VISUAL ARTIFACTS IN outputs/:
--------------------------------------------------------------------------------
1. outputs/01_selling_price_distribution.png
2. outputs/02_selling_price_boxplot.png
3. outputs/03_car_age_vs_selling_price.png
4. outputs/04_kms_driven_vs_selling_price.png
5. outputs/05_present_price_vs_selling_price.png
6. outputs/06_categorical_vs_selling_price.png
7. outputs/07_correlation_heatmap.png
================================================================================
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n[SUCCESS] EDA report saved to: {report_path}")


def main():
    print("=" * 70)
    print("USED-CAR PRICE PREDICTION - EXPLORATORY DATA ANALYSIS (MODULE 3)")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "cleaned_car_data.csv")
    output_dir = os.path.join(base_dir, "outputs")
    report_path = os.path.join(output_dir, "eda_report.txt")

    # Step 1: Load Cleaned Dataset
    df = load_dataset(data_path)

    # Step 2: Display Summary Information to Terminal
    print("\n" + "-" * 40)
    print("1. DATASET SHAPE & ATTRIBUTES:")
    print(f"   - Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   - Columns: {list(df.columns)}")
    print(f"   - Data Types:\n{df.dtypes.to_string()}")

    print("\n" + "-" * 40)
    print("2. TARGET VARIABLE STATISTICS (Selling_Price in Lakhs INR):")
    sp = df['Selling_Price']
    print(f"   - Mean:    Rs. {sp.mean():.2f} Lakhs")
    print(f"   - Median:  Rs. {sp.median():.2f} Lakhs")
    print(f"   - Std Dev: Rs. {sp.std():.2f} Lakhs")
    print(f"   - Min:     Rs. {sp.min():.2f} Lakhs")
    print(f"   - 25% Q1:  Rs. {sp.quantile(0.25):.2f} Lakhs")
    print(f"   - 75% Q3:  Rs. {sp.quantile(0.75):.2f} Lakhs")
    print(f"   - Max:     Rs. {sp.max():.2f} Lakhs")
    print(f"   - Skewness: {sp.skew():.2f}")

    print("\n" + "-" * 40)
    print("3. NUMERICAL ATTRIBUTES SUMMARY:")
    print(df.describe().round(2).to_string())

    # Step 3: Generate and Save Visualizations
    generate_plots(df, output_dir)

    # Step 4: Write EDA Report
    generate_eda_report(df, report_path)

    print("\n" + "=" * 70)
    print("[COMPLETED] Module 3 Exploratory Data Analysis executed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
