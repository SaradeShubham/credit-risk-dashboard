# preprocessing
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def load_raw_data():
    
    df = pd.read_csv("C:\Users\shubh\OneDrive\Desktop\Home_Credit_Dashboard\application_train.csv")
    print("Download complete.")
    return df

def preprocess_data(df):
    """The complete, optimized preprocessing function."""
    print("Step 2: Starting data cleaning and feature engineering...")
    df_copy = df.copy()

    # Feature Engineering
    new_cols = {
        'AGE_YEARS': -df_copy['DAYS_BIRTH'] / 365.25,
        'EMPLOYMENT_YEARS': -df_copy['DAYS_EMPLOYED'] / 365.25
    }
    df_copy = df_copy.assign(**new_cols)
    df_copy['EMPLOYMENT_YEARS'] = df_copy['EMPLOYMENT_YEARS'].replace(365243 / -365.25, np.nan)
    df_copy['DTI'] = df_copy['AMT_ANNUITY'] / df_copy['AMT_INCOME_TOTAL']
    df_copy['LOAN_TO_INCOME'] = df_copy['AMT_CREDIT'] / df_copy['AMT_INCOME_TOTAL']
    df_copy['ANNUITY_TO_CREDIT'] = df_copy['AMT_ANNUITY'] / df_copy['AMT_CREDIT']
    
    # Outlier Handling
    skewed_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
    for col in skewed_cols:
        if col in df_copy.columns:
            df_copy[col] = winsorize(df_copy[col].astype(float), limits=[0.01, 0.01])

    # Rare Category Consolidation
    for col in df_copy.select_dtypes(include='object').columns:
        category_freq = df_copy[col].value_counts(normalize=True)
        rare_categories = category_freq[category_freq < 0.01].index
        if len(rare_categories) > 0:
            df_copy[col] = df_copy[col].replace(rare_categories, 'Other')

    # Missing Values
    missing_percent = df_copy.isnull().sum() / len(df_copy) * 100
    cols_to_drop = missing_percent[missing_percent > 60].index
    df_copy.drop(columns=cols_to_drop, inplace=True)
    
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
        else:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            
    # Income Brackets
    df_copy['INCOME_BRACKET'] = pd.qcut(df_copy['AMT_INCOME_TOTAL'], 
                                   q=[0, 0.25, 0.75, 1.0], 
                                   labels=['Low', 'Mid', 'High'],
                                   duplicates='drop')
    
    print("Preprocessing complete.")
    return df_copy.copy()

# *** Main execution block ===
if __name__ == "__main__":
    raw_data = load_raw_data()
    cleaned_data = preprocess_data(raw_data)
    
    # Save the clean data to a fast Parquet file
    output_filename = 'cleaned_application_data.parquet'
    cleaned_data.to_parquet(output_filename, index=False)
    print(f"\n✅ Success! Cleaned data saved to '{output_filename}'.")
    print("You can now upload this file to your GitHub dataset repository.")