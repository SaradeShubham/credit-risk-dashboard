# utils
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

# ***HELPER FUNCTION
def reset_filters():
    """Resets all the filter values stored in the session state."""
    filter_keys = ['age_range_filter', 'CODE_GENDER_filter', 'NAME_FAMILY_STATUS_filter', 
                   'NAME_EDUCATION_TYPE_filter', 'NAME_HOUSING_TYPE_filter', 'INCOME_BRACKET_filter']
    for key in filter_keys:
        if key in st.session_state:
            del st.session_state[key]

@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')

#**** DATA PREPROCESSING FUNCTION****
# We are adding this function back to process the raw CSV.
def preprocess_data(df):
    """Takes the raw dataframe and returns a cleaned, ready-to-use version."""
    df_copy = df.copy()

    # Feature Engineering
    df_copy = df_copy.assign(
        AGE_YEARS = -df_copy['DAYS_BIRTH'] / 365.25,
        EMPLOYMENT_YEARS = -df_copy['DAYS_EMPLOYED'] / 365.25
    )
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
    return df_copy.copy()

# == DATA LOADING==
@st.cache_data
def load_data():
    """Loads the local raw CSV and preprocesses it."""
    with st.spinner("Loading and preprocessing local data..."):
        # Step 1: Read the local raw CSV file
        df = pd.read_csv('application_train.csv')
        # Step 2: Clean and preprocess the data
        cleaned_df = preprocess_data(df)
    return cleaned_df

# ==SIDEBAR AND FILTERING==
def create_sidebar_filters(df):
    """Creates all global sidebar filters and returns the filtered dataframe."""
    st.sidebar.header("Global Filters")

    # Session State Initialization
    if 'age_range_filter' not in st.session_state:
        min_age, max_age = int(df['AGE_YEARS'].min()), int(df['AGE_YEARS'].max())
        st.session_state.age_range_filter = (min_age, max_age)

    categorical_filters = ['CODE_GENDER', 'NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE', 'INCOME_BRACKET']
    for column in categorical_filters:
        filter_key = f"{column}_filter"
        if filter_key not in st.session_state:
            st.session_state[filter_key] = df[column].unique().tolist()

    # Filter Widgets
    st.sidebar.slider("Age Range", int(df['AGE_YEARS'].min()), int(df['AGE_YEARS'].max()), key='age_range_filter')

    for column in categorical_filters:
        label = column.replace('_', ' ').title()
        options = df[column].unique().tolist()
        st.sidebar.multiselect(label, options, key=f"{column}_filter")

    # Applying filters
    filtered_df = df[
        (df['AGE_YEARS'].between(*st.session_state.age_range_filter)) &
        (df['CODE_GENDER'].isin(st.session_state.CODE_GENDER_filter)) &
        (df['NAME_FAMILY_STATUS'].isin(st.session_state.NAME_FAMILY_STATUS_filter)) &
        (df['NAME_EDUCATION_TYPE'].isin(st.session_state.NAME_EDUCATION_TYPE_filter)) &
        (df['NAME_HOUSING_TYPE'].isin(st.session_state.NAME_HOUSING_TYPE_filter)) &
        (df['INCOME_BRACKET'].isin(st.session_state.INCOME_BRACKET_filter))
    ]
    
    st.sidebar.markdown("---") 

    # Reset and Download Buttons
    st.sidebar.button("Reset All Filters", on_click=reset_filters, width='stretch')
    
    csv_data = convert_df_to_csv(filtered_df)
    st.sidebar.download_button(
       label="📥 Download Filtered Data",
       data=csv_data,
       file_name='filtered_credit_data.csv',
       mime='text/csv',
       width='stretch'
    )
    
    return filtered_df