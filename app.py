# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize



# Set the page to a wide layout
st.set_page_config(layout="wide")

# --- Data Loading and Caching ---
@st.cache_data
def load_data():
    """Loads data from the permanent GitHub LFS link."""
    # Paste the new URL you copied from the GitHub "Download" button here
    url = 'https://github.com/vishnu-narayan-rgb/my-datasets/raw/refs/heads/master/application_train.csv?download=' 
    
    with st.spinner("Downloading data from remote source... this may take a moment on first run."):
        df = pd.read_csv(url)
    return df



# --- Data Preprocessing Function ---
def preprocess_data(df):
    """
    Takes the raw dataframe and returns a cleaned, ready-to-use version
    with optimized performance to prevent fragmentation.
    """
    df_copy = df.copy()

    # --- Create new columns in a dictionary first ---
    new_cols = {
        'AGE_YEARS': -df_copy['DAYS_BIRTH'] / 365.25,
        'EMPLOYMENT_YEARS': -df_copy['DAYS_EMPLOYED'] / 365.25
    }
    
    # --- Assign new columns all at once to be more efficient ---
    df_copy = df_copy.assign(**new_cols)

    # Handle special value in the new employment column
    df_copy['EMPLOYMENT_YEARS'] = df_copy['EMPLOYMENT_YEARS'].replace(365243 / -365.25, np.nan)
    
    # --- Create financial ratio columns ---
    df_copy['DTI'] = df_copy['AMT_ANNUITY'] / df_copy['AMT_INCOME_TOTAL']
    df_copy['LOAN_TO_INCOME'] = df_copy['AMT_CREDIT'] / df_copy['AMT_INCOME_TOTAL']
    df_copy['ANNUITY_TO_CREDIT'] = df_copy['AMT_ANNUITY'] / df_copy['AMT_CREDIT']
    
    # --- Advanced: Outlier Handling (Winsorizing) ---
    skewed_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
    for col in skewed_cols:
        if col in df_copy.columns:
            df_copy[col] = winsorize(df_copy[col].astype(float), limits=[0.01, 0.01])

    # --- Advanced: Rare Category Consolidation ---
    for col in df_copy.select_dtypes(include='object').columns:
        category_freq = df_copy[col].value_counts(normalize=True)
        rare_categories = category_freq[category_freq < 0.01].index
        if len(rare_categories) > 0:
            df_copy[col] = df_copy[col].replace(rare_categories, 'Other')

    # --- Handle Missing Values ---
    missing_percent = df_copy.isnull().sum() / len(df_copy) * 100
    cols_to_drop = missing_percent[missing_percent > 60].index
    df_copy.drop(columns=cols_to_drop, inplace=True)
    
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
        else:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            
    # --- Create Income Brackets ---
    # We add duplicates='drop' to prevent errors if bin edges are not unique
    df_copy['INCOME_BRACKET'] = pd.qcut(df_copy['AMT_INCOME_TOTAL'], 
                                   q=[0, 0.25, 0.75, 1.0], 
                                   labels=['Low', 'Mid', 'High'],
                                   duplicates='drop')
    
    # --- Final De-fragmentation ---
    # As suggested by the warning, this creates a new, memory-efficient copy.
    return df_copy.copy()
    
@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')

def reset_filters():
    """Resets all the filter values stored in the session state."""
    for key in st.session_state.keys():
        if key.endswith('_filter') or key == 'age_range_filter':
            del st.session_state[key]
# =====================================================================================
# --- Main App Execution Flow ---
# =====================================================================================

# Load and preprocess data
raw_df = load_data()
cleaned_df = preprocess_data(raw_df)

# --- Sidebar for Navigation and Filters ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "Target & Risk Segmentation", 
                                  "Demographics & Household", "Financial Health", 
                                  "Correlations & Drivers"])

st.sidebar.header("Global Filters")

# --- NEW: Session State Initialization for Filters ---
if 'age_range_filter' not in st.session_state:
    min_age, max_age = int(cleaned_df['AGE_YEARS'].min()), int(cleaned_df['AGE_YEARS'].max())
    st.session_state.age_range_filter = (min_age, max_age)

categorical_filters = ['CODE_GENDER', 'NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE', 'INCOME_BRACKET']
for column in categorical_filters:
    filter_key = f"{column}_filter"
    if filter_key not in st.session_state:
        st.session_state[filter_key] = cleaned_df[column].unique().tolist()

# --- NEW: Filter Widgets using Session State ---
age_range = st.sidebar.slider("Age Range", 
                              int(cleaned_df['AGE_YEARS'].min()), 
                              int(cleaned_df['AGE_YEARS'].max()), 
                              value=st.session_state.age_range_filter, 
                              key='age_range_filter')

for column in categorical_filters:
    label = column.replace('_', ' ').title()
    options = cleaned_df[column].unique().tolist()
    st.sidebar.multiselect(label, options, key=f"{column}_filter")

# --- NEW: Reset and Download Buttons ---
st.sidebar.button("Reset All Filters", on_click=reset_filters, use_container_width=True)
filtered_csv = convert_df_to_csv(cleaned_df[  # Apply filters here just for download
    (cleaned_df['AGE_YEARS'].between(*st.session_state.age_range_filter)) &
    (cleaned_df['CODE_GENDER'].isin(st.session_state.CODE_GENDER_filter)) &
    (cleaned_df['NAME_FAMILY_STATUS'].isin(st.session_state.NAME_FAMILY_STATUS_filter)) &
    (cleaned_df['NAME_EDUCATION_TYPE'].isin(st.session_state.NAME_EDUCATION_TYPE_filter)) &
    (cleaned_df['NAME_HOUSING_TYPE'].isin(st.session_state.NAME_HOUSING_TYPE_filter)) &
    (cleaned_df['INCOME_BRACKET'].isin(st.session_state.INCOME_BRACKET_filter))
])
st.sidebar.download_button(
   label="📥 Download Filtered Data as CSV",
   data=filtered_csv,
   file_name='filtered_home_credit_data.csv',
   mime='text/csv',
   use_container_width=True
)


# --- Apply filters to create the final dataframe for display ---
filtered_df = cleaned_df[
    (cleaned_df['AGE_YEARS'].between(*st.session_state.age_range_filter)) &
    (cleaned_df['CODE_GENDER'].isin(st.session_state.CODE_GENDER_filter)) &
    (cleaned_df['NAME_FAMILY_STATUS'].isin(st.session_state.NAME_FAMILY_STATUS_filter)) &
    (cleaned_df['NAME_EDUCATION_TYPE'].isin(st.session_state.NAME_EDUCATION_TYPE_filter)) &
    (cleaned_df['NAME_HOUSING_TYPE'].isin(st.session_state.NAME_HOUSING_TYPE_filter)) &
    (cleaned_df['INCOME_BRACKET'].isin(st.session_state.INCOME_BRACKET_filter))
]
# 4. Display the selected page content
# The content of each page will be built using the `filtered_df`.
if page == "Project Overview":
    st.title("📊 Page 1: Overview & Data Quality")
    st.write(f"Displaying data for **{filtered_df.shape[0]}** applicants based on filters.")
    # All KPIs and graphs for Page 1 will go here.
    # --- KPIs ---
    # --- KPIs for Overview (10 KPIs) ---
    # --- KPIs for Overview (10 KPIs) ---
    st.header("Key Performance Indicators")

    # --- Row 1: Portfolio Overview ---
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    # KPI 1: Total Applicants
    total_applicants = filtered_df['SK_ID_CURR'].nunique()
    kpi1.metric(label="Total Applicants 👥", value=f"{total_applicants:,}")

    # KPI 2: Default Rate
    default_rate = filtered_df['TARGET'].mean() * 100
    kpi2.metric(label="Default Rate 👎", value=f"{default_rate:.2f}%")

    # KPI 3: Repaid Rate
    repaid_rate = (1 - filtered_df['TARGET'].mean()) * 100
    kpi3.metric(label="Repaid Rate 👍", value=f"{repaid_rate:.2f}%")
    
    # KPI 4: Median Applicant Age
    median_age = filtered_df['AGE_YEARS'].median()
    kpi4.metric(label="Median Applicant Age", value=f"{median_age:.1f} Yrs")

    # KPI 5: Average Credit Amount
    avg_credit = filtered_df['AMT_CREDIT'].mean()
    kpi5.metric(label="Average Credit Amount 💳", value=f"₹{avg_credit:,.0f}")
    
    st.markdown("<br>", unsafe_allow_html=True) # Adds a little space between rows

    # --- Row 2: Data Quality & Financials ---
    kpi6, kpi7, kpi8, kpi9, kpi10 = st.columns(5)
    
    # KPI 6: Total Features
    total_features = filtered_df.shape[1]
    kpi6.metric(label="Total Features (Columns) 📊", value=total_features)

    # KPI 7: Numerical Features
    num_features = filtered_df.select_dtypes(include=np.number).shape[1]
    kpi7.metric(label="Numerical Features", value=num_features)

    # KPI 8: Categorical Features
    cat_features = filtered_df.select_dtypes(include=['object', 'category']).shape[1]
    kpi8.metric(label="Categorical Features", value=cat_features)
    
    # KPI 9: Features with Missing Values (Post-Imputation Check)
    missing_features = filtered_df.isnull().sum().gt(0).sum()
    kpi9.metric(label="Features w/ Missing Values", value=missing_features, help="This should be 0 after our preprocessing.")

    # KPI 10: Median Annual Income --- THIS LINE IS NOW CORRECTED ---
    median_income = filtered_df['AMT_INCOME_TOTAL'].median()
    kpi10.metric(label="Median Annual Income 💵", value=f"₹{median_income:,.0f}")
    
    # --- Graphs ---
    # --- Graphs ---
    st.header("Visualizations of Key Features")
    
    # --- Row 1: Loan Status & Gender ---
    graph_col1, graph_col2 = st.columns(2)
    
    with graph_col1:
        st.subheader("Loan Status Distribution")
        fig, ax = plt.subplots(figsize=(6, 5))
        target_counts = filtered_df['TARGET'].value_counts()
        ax.pie(target_counts, labels=['Repaid (0)', 'Default (1)'], autopct='%1.1f%%', 
               startangle=90, colors=['#66b3ff','#ff9999'])
        ax.axis('equal')
        st.pyplot(fig)

    with graph_col2:
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.countplot(x='CODE_GENDER', data=filtered_df, ax=ax, palette='viridis', order=filtered_df['CODE_GENDER'].value_counts().index)
        ax.set_xlabel("Gender")
        ax.set_ylabel("Number of Applicants")
        st.pyplot(fig)

    # --- Row 2: Age Distribution ---
    st.subheader("Applicant Age Distribution")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(filtered_df['AGE_YEARS'], bins=30, kde=True, ax=ax, color='skyblue')
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # --- Row 3: Income and Credit Histograms ---
    graph_col3, graph_col4 = st.columns(2)

    with graph_col3:
        st.subheader("Total Income Distribution")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.histplot(filtered_df['AMT_INCOME_TOTAL'], bins=30, kde=False, ax=ax, color='salmon')
        ax.set_xlabel("Total Income")
        ax.set_ylabel("Frequency")
        ax.ticklabel_format(style='plain', axis='x')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with graph_col4:
        st.subheader("Credit Amount Distribution")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.histplot(filtered_df['AMT_CREDIT'], bins=30, kde=False, ax=ax, color='lightgreen')
        ax.set_xlabel("Credit Amount")
        ax.set_ylabel("Frequency")
        ax.ticklabel_format(style='plain', axis='x')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    # --- Row 4: Income and Credit Boxplots ---
    graph_col5, graph_col6 = st.columns(2)

    with graph_col5:
        st.subheader("Total Income Boxplot")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(y='AMT_INCOME_TOTAL', data=filtered_df, ax=ax, color='salmon')
        ax.set_ylabel("Total Income")
        st.pyplot(fig)
    
    with graph_col6:
        st.subheader("Credit Amount Boxplot")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(y='AMT_CREDIT', data=filtered_df, ax=ax, color='lightgreen')
        ax.set_ylabel("Credit Amount")
        st.pyplot(fig)

    # --- Row 5: Categorical Distributions ---
    graph_col7, graph_col8 = st.columns(2)
    
    with graph_col7:
        st.subheader("Family Status")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.countplot(y='NAME_FAMILY_STATUS', data=filtered_df, ax=ax, palette='plasma', hue='NAME_FAMILY_STATUS', legend=False, order=filtered_df['NAME_FAMILY_STATUS'].value_counts().index)
        ax.set_xlabel("Number of Applicants")
        ax.set_ylabel("Family Status")
        st.pyplot(fig)

    with graph_col8:
        st.subheader("Education Type")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.countplot(y='NAME_EDUCATION_TYPE', data=filtered_df, ax=ax, palette='magma', order=filtered_df['NAME_EDUCATION_TYPE'].value_counts().index)
        ax.set_xlabel("Number of Applicants")
        ax.set_ylabel("Education Type")
        st.pyplot(fig)
    
    # --- Row 6: Housing Type (The 10th Graph) ---
    st.subheader("Housing Type Distribution")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.countplot(y='NAME_HOUSING_TYPE', data=filtered_df, ax=ax, palette='cividis', order=filtered_df['NAME_HOUSING_TYPE'].value_counts().index)
    ax.set_xlabel("Number of Applicants")
    ax.set_ylabel("Housing Type")
    st.pyplot(fig)



elif page == "Target & Risk Segmentation":
    st.title("🎯 Page 2: Target & Risk Segmentation")
    st.write(f"Displaying data for **{filtered_df.shape[0]}** applicants based on filters.")

    # --- KPIs for Risk Segmentation (10 KPIs)---
    st.header("Key Risk Indicators")

    # Create dataframes for defaulters and non-defaulters for easy comparison
    defaulters_df = filtered_df[filtered_df['TARGET'] == 1]
    non_defaulters_df = filtered_df[filtered_df['TARGET'] == 0]
    
    # --- Row 1 of KPIs ---
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    kpi1.metric(label="Applicants in Selection 👥", value=f"{len(filtered_df):,}")
    kpi2.metric(label="Total Defaults 📉", value=f"{len(defaulters_df):,}")
    kpi3.metric(label="Overall Default Rate %", value=f"{filtered_df['TARGET'].mean() * 100:.2f}%")
    kpi4.metric(label="Avg Age (Defaulters)", value=f"{defaulters_df['AGE_YEARS'].mean():.1f}")
    kpi5.metric(label="Avg Employment Yrs (Defaulters)", value=f"{defaulters_df['EMPLOYMENT_YEARS'].mean():.1f}")
    
    # --- Row 2 of KPIs ---
    kpi6, kpi7, kpi8, kpi9, kpi10 = st.columns(5)

    avg_income_def = defaulters_df['AMT_INCOME_TOTAL'].mean()
    avg_income_non_def = non_defaulters_df['AMT_INCOME_TOTAL'].mean()
    kpi6.metric(label="Avg Income (Defaulters) 💰", value=f"₹{avg_income_def:,.0f}", 
                delta=f"₹{avg_income_def - avg_income_non_def:,.0f} vs. Repaid")

    avg_credit_def = defaulters_df['AMT_CREDIT'].mean()
    avg_credit_non_def = non_defaulters_df['AMT_CREDIT'].mean()
    kpi7.metric(label="Avg Credit (Defaulters) 💳", value=f"₹{avg_credit_def:,.0f}", 
                delta=f"₹{avg_credit_def - avg_credit_non_def:,.0f} vs. Repaid")
    
    avg_annuity_def = defaulters_df['AMT_ANNUITY'].mean()
    avg_annuity_non_def = non_defaulters_df['AMT_ANNUITY'].mean()
    kpi8.metric(label="Avg Annuity (Defaulters)", value=f"₹{avg_annuity_def:,.0f}",
                delta=f"₹{avg_annuity_def - avg_annuity_non_def:,.0f} vs. Repaid")
    
    dti_def = defaulters_df['DTI'].mean()
    kpi9.metric(label="Avg DTI (Defaulters)", value=f"{dti_def:.2%}")

    lti_def = defaulters_df['LOAN_TO_INCOME'].mean()
    kpi10.metric(label="Avg Loan-to-Income (Defaulters)", value=f"{lti_def:.2f}")

    # --- Graphs for Risk Segmentation (10 Graphs) ---
    st.header("Visual Risk Segmentation")
    
    graph_col1, graph_col2 = st.columns(2)

    with graph_col1:
        st.subheader("Default vs. Repaid Counts")
        fig, ax = plt.subplots()
        sns.countplot(x='TARGET', data=filtered_df, ax=ax, palette=['#66b3ff','#ff9999'])
        ax.set_xlabel("Loan Status (1 = Default, 0 = Repaid)")
        ax.set_ylabel("Number of Applicants")
        st.pyplot(fig)

    with graph_col2:
        st.subheader("Default Rate by Gender")
        gender_default_rate = filtered_df.groupby('CODE_GENDER')['TARGET'].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(x='CODE_GENDER', y='TARGET', data=gender_default_rate, ax=ax, palette='viridis')
        ax.set_xlabel("Gender")
        ax.set_ylabel("Default Rate")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        st.pyplot(fig)

    graph_col3, graph_col4 = st.columns(2)

    with graph_col3:
        st.subheader("Default Rate by Education")
        edu_default_rate = filtered_df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().sort_values(ascending=False).reset_index()
        fig, ax = plt.subplots()
        sns.barplot(y='NAME_EDUCATION_TYPE', x='TARGET', data=edu_default_rate, ax=ax, palette='plasma')
        ax.set_xlabel("Default Rate")
        ax.set_ylabel("Education Type")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        st.pyplot(fig)

    with graph_col4:
        st.subheader("Default Rate by Family Status")
        family_default_rate = filtered_df.groupby('NAME_FAMILY_STATUS')['TARGET'].mean().sort_values(ascending=False).reset_index()
        fig, ax = plt.subplots()
        sns.barplot(y='NAME_FAMILY_STATUS', x='TARGET', data=family_default_rate, ax=ax, palette='magma')
        ax.set_xlabel("Default Rate")
        ax.set_ylabel("Family Status")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        st.pyplot(fig)
        
    graph_col5, graph_col6 = st.columns(2)

    with graph_col5:
        st.subheader("Income Distribution by Loan Status")
        fig, ax = plt.subplots()
        sns.boxplot(x='TARGET', y='AMT_INCOME_TOTAL', data=filtered_df, ax=ax, palette=['#66b3ff','#ff9999'])
        ax.set_xlabel("Loan Status")
        ax.set_ylabel("Total Income")
        ax.set_yscale('log')
        st.pyplot(fig)

    with graph_col6:
        st.subheader("Credit Amount by Loan Status")
        fig, ax = plt.subplots()
        sns.boxplot(x='TARGET', y='AMT_CREDIT', data=filtered_df, ax=ax, palette=['#66b3ff','#ff9999'])
        ax.set_xlabel("Loan Status")
        ax.set_ylabel("Credit Amount")
        ax.set_yscale('log')
        st.pyplot(fig)
        
    graph_col7, graph_col8 = st.columns(2)
    
    with graph_col7:
        st.subheader("Age Distribution by Loan Status")
        fig, ax = plt.subplots()
        sns.violinplot(x='TARGET', y='AGE_YEARS', data=filtered_df, ax=ax, palette=['#66b3ff','#ff9999'])
        ax.set_xlabel("Loan Status")
        ax.set_ylabel("Age (Years)")
        st.pyplot(fig)
        
    with graph_col8:
        st.subheader("Employment Years by Loan Status")
        fig, ax = plt.subplots()
        sns.histplot(data=filtered_df, x='EMPLOYMENT_YEARS', hue='TARGET', multiple='stack', 
                     ax=ax, palette=['#66b3ff','#ff9999'], bins=30)
        ax.set_xlabel("Employment (Years)")
        ax.set_ylabel("Number of Applicants")
        st.pyplot(fig)

    graph_col9, graph_col10 = st.columns(2)

    with graph_col9:
        st.subheader("Default Rate by Contract Type")
        contract_default_rate = filtered_df.groupby('NAME_CONTRACT_TYPE')['TARGET'].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(x='NAME_CONTRACT_TYPE', y='TARGET', data=contract_default_rate, ax=ax, palette='viridis')
        ax.set_xlabel("Contract Type")
        ax.set_ylabel("Default Rate")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        st.pyplot(fig)

    with graph_col10:
        st.subheader("Default Rate by Housing Type")
        housing_default_rate = filtered_df.groupby('NAME_HOUSING_TYPE')['TARGET'].mean().sort_values(ascending=False).reset_index()
        fig, ax = plt.subplots()
        sns.barplot(y='NAME_HOUSING_TYPE', x='TARGET', data=housing_default_rate, ax=ax, palette='plasma')
        ax.set_xlabel("Default Rate")
        ax.set_ylabel("Housing Type")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        st.pyplot(fig)

    # --- Narrative Insights ---
    st.header("Insights on Risk Segmentation")
    st.markdown("""
    * **Highest Risk Segments:** The bar charts consistently highlight specific groups with higher-than-average default rates. Applicants who are single, have 'Secondary' education, and live in 'Rented apartments' or 'With parents' tend to show a higher propensity to default.
    * **Lowest Risk Segments:** Conversely, applicants who are 'Married', have 'Higher education', and are 'House / apartment' owners exhibit the lowest default rates. This suggests financial stability and life stage are strong indicators of repayment ability.
    * **Financial Indicators:** While the income distributions overlap, the KPI `delta` shows that defaulters, on average, have slightly lower incomes and request slightly higher credit amounts compared to those who repay. The violin plot for age shows that younger applicants have a higher concentration in the defaulter group.
    """)


elif page == "Demographics & Household":
    st.title("👨‍👩‍👧‍👦 Page 3: Demographics & Household Profile")
    st.write(f"Displaying data for **{filtered_df.shape[0]}** applicants based on filters.")

    # --- KPIs for Demographics (10 KPIs) ---
    st.header("Key Demographic Indicators")

    # Reuse defaulter/non-defaulter dataframes from previous step
    defaulters_df = filtered_df[filtered_df['TARGET'] == 1]
    non_defaulters_df = filtered_df[filtered_df['TARGET'] == 0]

    # --- Row 1 of KPIs ---
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    gender_dist = filtered_df['CODE_GENDER'].value_counts(normalize=True) * 100
    kpi1.metric(label="% Female Applicants 👩", value=f"{gender_dist.get('F', 0):.1f}%")
    kpi2.metric(label="% Male Applicants 👨", value=f"{gender_dist.get('M', 0):.1f}%")
    
    kpi3.metric(label="Avg Age (Defaulters)", value=f"{defaulters_df['AGE_YEARS'].mean():.1f}")
    kpi4.metric(label="Avg Age (Repaid)", value=f"{non_defaulters_df['AGE_YEARS'].mean():.1f}")
    
    avg_fam_size = filtered_df['CNT_FAM_MEMBERS'].mean()
    kpi5.metric(label="Avg Family Size 👨‍👩‍👧", value=f"{avg_fam_size:.2f}")

    # --- Row 2 of KPIs ---
    kpi6, kpi7, kpi8, kpi9, kpi10 = st.columns(5)
    
    pct_with_children = (filtered_df['CNT_CHILDREN'] > 0).mean() * 100
    kpi6.metric(label="% With Children", value=f"{pct_with_children:.1f}%")
    
    pct_married = (filtered_df['NAME_FAMILY_STATUS'] == 'Married').mean() * 100
    kpi7.metric(label="% Married", value=f"{pct_married:.1f}%")

    pct_higher_edu = (filtered_df['NAME_EDUCATION_TYPE'] == 'Higher education').mean() * 100
    kpi8.metric(label="% Higher Education 🎓", value=f"{pct_higher_edu:.1f}%")

    pct_with_parents = (filtered_df['NAME_HOUSING_TYPE'] == 'With parents').mean() * 100
    kpi9.metric(label="% Living With Parents", value=f"{pct_with_parents:.1f}%")

    avg_emp_years = filtered_df['EMPLOYMENT_YEARS'].mean()
    kpi10.metric(label="Avg Employment Years 🗓️", value=f"{avg_emp_years:.1f}")
    
    # --- Graphs for Demographics (10 Graphs) ---
    st.header("Visual Demographic Profiles")

    # --- Row 1: Age Distributions ---
    graph_col1, graph_col2 = st.columns(2)
    with graph_col1:
        st.subheader("Age Distribution (All Applicants)")
        fig, ax = plt.subplots()
        sns.histplot(data=filtered_df, x='AGE_YEARS', kde=True, ax=ax, color='teal')
        ax.set_xlabel("Age (Years)")
        st.pyplot(fig)
    
    with graph_col2:
        st.subheader("Age Distribution by Loan Status")
        fig, ax = plt.subplots()
        sns.histplot(data=filtered_df, x='AGE_YEARS', hue='TARGET', kde=True, ax=ax, palette=['#66b3ff','#ff9999'])
        ax.set_xlabel("Age (Years)")
        st.pyplot(fig)

    # --- Row 2: Gender & Family Status ---
    graph_col3, graph_col4 = st.columns(2)
    with graph_col3:
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots()
        filtered_df['CODE_GENDER'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=['#ff9999','#66b3ff','#99ff99'])
        ax.set_ylabel('') # Hide the y-label
        st.pyplot(fig)

    with graph_col4:
        st.subheader("Family Status Distribution")
        fig, ax = plt.subplots()
        sns.countplot(y='NAME_FAMILY_STATUS', data=filtered_df, ax=ax, palette='viridis', order=filtered_df['NAME_FAMILY_STATUS'].value_counts().index)
        ax.set_xlabel("Number of Applicants")
        ax.set_ylabel("Family Status")
        st.pyplot(fig)
        
    # --- Row 3: Education & Housing ---
    graph_col5, graph_col6 = st.columns(2)
    with graph_col5:
        st.subheader("Education Distribution")
        fig, ax = plt.subplots()
        sns.countplot(y='NAME_EDUCATION_TYPE', data=filtered_df, ax=ax, palette='plasma', order=filtered_df['NAME_EDUCATION_TYPE'].value_counts().index)
        ax.set_xlabel("Number of Applicants")
        ax.set_ylabel("Education Type")
        st.pyplot(fig)
        
    with graph_col6:
        st.subheader("Housing Type Distribution")
        fig, ax = plt.subplots()
        filtered_df['NAME_HOUSING_TYPE'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, wedgeprops=dict(width=0.3))
        ax.set_ylabel('')
        st.pyplot(fig) # Donut chart
        
    # --- Row 4: Children Count & Occupation ---
    graph_col7, graph_col8 = st.columns(2)
    with graph_col7:
        st.subheader("Number of Children")
        fig, ax = plt.subplots()
        sns.countplot(x='CNT_CHILDREN', data=filtered_df, ax=ax, palette='magma')
        ax.set_xlabel("Number of Children")
        ax.set_ylabel("Number of Applicants")
        st.pyplot(fig)
    
    with graph_col8:
        st.subheader("Top 10 Occupations")
        fig, ax = plt.subplots()
        top_10_occupations = filtered_df['OCCUPATION_TYPE'].value_counts().nlargest(10)
        sns.barplot(y=top_10_occupations.index, x=top_10_occupations.values, ax=ax, palette='cividis')
        ax.set_xlabel("Number of Applicants")
        ax.set_ylabel("Occupation Type")
        st.pyplot(fig)
        
    # --- Row 5: Age vs Target & Correlation Heatmap ---
    graph_col9, graph_col10 = st.columns(2)
    with graph_col9:
        st.subheader("Age vs. Loan Status")
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered_df, x='TARGET', y='AGE_YEARS', ax=ax, palette=['#66b3ff','#ff9999'])
        ax.set_xlabel("Loan Status (1=Default, 0=Repaid)")
        ax.set_ylabel("Age (Years)")
        st.pyplot(fig)
        
    with graph_col10:
        st.subheader("Demographic Correlation Heatmap")
        fig, ax = plt.subplots()
        corr_cols = ['AGE_YEARS', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'TARGET']
        corr_matrix = filtered_df[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
        st.pyplot(fig)

    # --- Narrative Insights ---
    st.header("Insights on Demographics")
    st.markdown("""
    * **Life Stage Matters:** The boxplot and age distribution graphs show that defaulters tend to be slightly younger than those who repay. This, combined with the higher default rates for those living with parents, suggests younger applicants in earlier life stages represent a higher risk.
    * **Family Structure & Stability:** Married applicants, who form the largest group, also have one of the lowest default rates. This indicates that family stability is correlated with financial reliability.
    * **Weak Correlations:** The heatmap shows that individual demographic factors like age, number of children, and family size have very weak direct correlations with defaulting. This implies that risk is not driven by a single demographic factor but likely a complex interaction between demographics, financial health, and loan characteristics.
    """)

elif page == "Financial Health":
    st.title("💰 Page 4: Financial Health & Affordability")
    st.write(f"Displaying data for **{filtered_df.shape[0]}** applicants based on filters.")

    # --- KPIs for Financial Health (10 KPIs) ---
    st.header("Key Financial Indicators")

    # Reuse defaulter/non-defaulter dataframes
    defaulters_df = filtered_df[filtered_df['TARGET'] == 1]
    non_defaulters_df = filtered_df[filtered_df['TARGET'] == 0]

    # --- Row 1 of KPIs ---
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    kpi1.metric(label="Avg Annual Income 💵", value=f"₹{filtered_df['AMT_INCOME_TOTAL'].mean():,.0f}")
    kpi2.metric(label="Median Annual Income", value=f"₹{filtered_df['AMT_INCOME_TOTAL'].median():,.0f}")
    kpi3.metric(label="Avg Credit Amount 💳", value=f"₹{filtered_df['AMT_CREDIT'].mean():,.0f}")
    kpi4.metric(label="Avg Loan Annuity", value=f"₹{filtered_df['AMT_ANNUITY'].mean():,.0f}")
    kpi5.metric(label="Avg Goods Price", value=f"₹{filtered_df['AMT_GOODS_PRICE'].mean():,.0f}")

    # --- Row 2 of KPIs ---
    kpi6, kpi7, kpi8, kpi9, kpi10 = st.columns(5)

    avg_dti = filtered_df['DTI'].mean()
    kpi6.metric(label="Avg Debt-to-Income (DTI)", value=f"{avg_dti:.2%}")

    avg_lti = filtered_df['LOAN_TO_INCOME'].mean()
    kpi7.metric(label="Avg Loan-to-Income (LTI)", value=f"{avg_lti:.2f}")

    income_gap = non_defaulters_df['AMT_INCOME_TOTAL'].mean() - defaulters_df['AMT_INCOME_TOTAL'].mean()
    kpi8.metric(label="Income Gap (Repaid - Default)", value=f"₹{income_gap:,.0f}")

    credit_gap = non_defaulters_df['AMT_CREDIT'].mean() - defaulters_df['AMT_CREDIT'].mean()
    kpi9.metric(label="Credit Gap (Repaid - Default)", value=f"₹{credit_gap:,.0f}")
    
    high_credit_pct = (filtered_df['AMT_CREDIT'] > 1_000_000).mean() * 100
    kpi10.metric(label="% High Credit Loans (>1M)", value=f"{high_credit_pct:.1f}%")

    # --- Graphs for Financial Health (10 Graphs) ---
    st.header("Visual Financial Analysis")

    # --- Row 1: Key Financial Distributions ---
    graph_col1, graph_col2, graph_col3 = st.columns(3)
    with graph_col1:
        st.subheader("Income Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['AMT_INCOME_TOTAL'], kde=True, ax=ax, color='blue', bins=30)
        ax.set_xlabel("Annual Income")
        st.pyplot(fig)
        
    with graph_col2:
        st.subheader("Credit Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['AMT_CREDIT'], kde=True, ax=ax, color='green', bins=30)
        ax.set_xlabel("Credit Amount")
        st.pyplot(fig)

    with graph_col3:
        st.subheader("Annuity Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['AMT_ANNUITY'], kde=True, ax=ax, color='red', bins=30)
        ax.set_xlabel("Loan Annuity")
        st.pyplot(fig)

    # --- Row 2: Relationships between Financial Variables ---
    graph_col4, graph_col5 = st.columns(2)
    with graph_col4:
        st.subheader("Income vs. Credit Amount")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='AMT_INCOME_TOTAL', y='AMT_CREDIT', data=filtered_df.sample(10000), 
                        ax=ax, alpha=0.3, hue='TARGET', palette=['#66b3ff','#ff9999'])
        ax.set_xlabel("Total Income")
        ax.set_ylabel("Credit Amount")
        ax.ticklabel_format(style='plain', axis='both')
        st.pyplot(fig)

    with graph_col5:
        st.subheader("Income vs. Annuity")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='AMT_INCOME_TOTAL', y='AMT_ANNUITY', data=filtered_df.sample(10000), 
                        ax=ax, alpha=0.3, hue='TARGET', palette=['#66b3ff','#ff9999'])
        ax.set_xlabel("Total Income")
        ax.set_ylabel("Annuity")
        ax.ticklabel_format(style='plain', axis='both')
        st.pyplot(fig)
        
    # --- Row 3: Financials by Loan Status (Boxplots) ---
    graph_col6, graph_col7 = st.columns(2)
    with graph_col6:
        st.subheader("Income by Loan Status")
        fig, ax = plt.subplots()
        sns.boxplot(x='TARGET', y='AMT_INCOME_TOTAL', data=filtered_df, ax=ax, palette=['#66b3ff','#ff9999'])
        ax.set_yscale('log')
        ax.set_xlabel("Loan Status (1=Default, 0=Repaid)")
        ax.set_ylabel("Total Income (Log Scale)")
        st.pyplot(fig)

    with graph_col7:
        st.subheader("Credit by Loan Status")
        fig, ax = plt.subplots()
        sns.boxplot(x='TARGET', y='AMT_CREDIT', data=filtered_df, ax=ax, palette=['#66b3ff','#ff9999'])
        ax.set_yscale('log')
        ax.set_xlabel("Loan Status (1=Default, 0=Repaid)")
        ax.set_ylabel("Credit Amount (Log Scale)")
        st.pyplot(fig)

    # --- Row 4: Default Rates by Financial Groups ---
    graph_col8, graph_col9 = st.columns(2)
    with graph_col8:
        st.subheader("Default Rate by Income Bracket")
        income_default_rate = filtered_df.groupby('INCOME_BRACKET')['TARGET'].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(x='INCOME_BRACKET', y='TARGET', data=income_default_rate, ax=ax, palette='coolwarm')
        ax.set_xlabel("Income Bracket")
        ax.set_ylabel("Default Rate")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        st.pyplot(fig)

    with graph_col9:
        st.subheader("Default Rate by LTI Bins")
        # Create temporary bins for Loan-to-Income ratio for visualization
        filtered_df['LTI_bins'] = pd.cut(filtered_df['LOAN_TO_INCOME'], bins=np.arange(0, 21, 2), right=False)
        lti_default_rate = filtered_df.groupby('LTI_bins')['TARGET'].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(x='LTI_bins', y='TARGET', data=lti_default_rate, ax=ax, palette='coolwarm_r')
        ax.set_xlabel("Loan-to-Income (LTI) Ratio")
        ax.set_ylabel("Default Rate")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    # --- Row 5: Financial Correlation Heatmap ---
    st.subheader("Financial Variable Correlations")
    fig, ax = plt.subplots(figsize=(10, 7))
    corr_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DTI', 'LOAN_TO_INCOME', 'TARGET']
    corr_matrix = filtered_df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt='.2f', ax=ax)
    st.pyplot(fig)

    # --- Narrative Insights ---
    st.header("Insights on Financial Health")
    st.markdown("""
    * **Affordability Thresholds:** The chart 'Default Rate by LTI Bins' is highly insightful. It clearly shows that as the Loan-to-Income (LTI) ratio increases—meaning the loan is a larger multiple of the applicant's annual income—the default rate rises significantly. This suggests a direct link between over-borrowing and defaulting.
    * **Income is a Key Differentiator:** The 'Default Rate by Income Bracket' shows a clear, inverse relationship: as income increases, the default rate decreases. The KPIs also highlight an "Income Gap," showing that applicants who repay their loans have a meaningfully higher income, on average, than those who default.
    * **Strong Correlations:** The final heatmap reveals expected strong positive correlations between credit amount, annuity, and the price of goods. More importantly, it shows a negative correlation between income and the DTI/LTI ratios, confirming that higher income individuals tend to have more manageable debt ratios.
    """)

elif page == "Correlations & Drivers":
    st.title("🔍 Page 5: Correlations, Drivers & Slice-and-Dice")
    st.write(f"Displaying data for **{filtered_df.shape[0]}** applicants based on filters.")

    # --- Correlation Calculation ---
    # Select only numeric columns for correlation analysis
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns
    corr_matrix = filtered_df[numeric_cols].corr()
    target_corr = corr_matrix['TARGET'].sort_values(ascending=False)

    # --- KPIs for Correlations (10 KPIs) ---
    st.header("Key Correlation Indicators")

    kpi_col1, kpi_col2 = st.columns(2)
    with kpi_col1:
        st.subheader("Top 5 Positive Correlations with Default")
        st.dataframe(target_corr.head(6)[1:]) # Show top 5 excluding TARGET itself

    with kpi_col2:
        st.subheader("Top 5 Negative Correlations with Default")
        st.dataframe(target_corr.tail(5))

    st.markdown("---")
    kpi3, kpi4, kpi5, kpi6, kpi7 = st.columns(5)
    
    corr_age_target = corr_matrix.loc['AGE_YEARS', 'TARGET']
    kpi3.metric(label="Corr(Age, TARGET)", value=f"{corr_age_target:.3f}")

    corr_emp_target = corr_matrix.loc['EMPLOYMENT_YEARS', 'TARGET']
    kpi4.metric(label="Corr(Employment, TARGET)", value=f"{corr_emp_target:.3f}")

    corr_inc_credit = corr_matrix.loc['AMT_INCOME_TOTAL', 'AMT_CREDIT']
    kpi5.metric(label="Corr(Income, Credit)", value=f"{corr_inc_credit:.3f}")
    
    most_corr_income = corr_matrix['AMT_INCOME_TOTAL'].sort_values(ascending=False).index[1]
    kpi6.metric(label="Most Correlated w/ Income", value=most_corr_income, help="Excluding itself")

    most_corr_credit = corr_matrix['AMT_CREDIT'].sort_values(ascending=False).index[1]
    kpi7.metric(label="Most Correlated w/ Credit", value=most_corr_credit, help="Excluding itself")


    # --- Graphs for Correlations (10 Graphs) ---
    st.header("Visual Correlation Analysis")

    # --- Row 1: Main Heatmap and Top Drivers Bar Chart ---
    graph_col1, graph_col2 = st.columns([2, 1]) # Make the heatmap wider
    with graph_col1:
        st.subheader("Correlation Heatmap of Key Features")
        fig, ax = plt.subplots(figsize=(12, 10))
        # Select a subset of important columns for a cleaner heatmap
        heatmap_cols = ['TARGET', 'AGE_YEARS', 'EMPLOYMENT_YEARS', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DTI', 'LOAN_TO_INCOME', 'CNT_CHILDREN']
        sns.heatmap(filtered_df[heatmap_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

    with graph_col2:
        st.subheader("Top Correlates with Default")
        # Get top 10 positive and negative correlates
        top_corr = pd.concat([target_corr.head(6)[1:], target_corr.tail(5)]).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 10))
        sns.barplot(x=top_corr.values, y=top_corr.index, ax=ax, palette='vlag')
        ax.set_xlabel("Correlation with TARGET")
        st.pyplot(fig)

    # --- Row 2: Interactive Scatter Plots ---
    graph_col3, graph_col4 = st.columns(2)
    with graph_col3:
        st.subheader("Age vs. Credit Amount")
        fig, ax = plt.subplots()
        sns.scatterplot(data=filtered_df.sample(2000), x='AGE_YEARS', y='AMT_CREDIT', hue='TARGET', palette=['#66b3ff','#ff9999'], alpha=0.5, ax=ax)
        ax.set_xlabel("Age (Years)")
        ax.set_ylabel("Credit Amount")
        st.pyplot(fig)

    with graph_col4:
        st.subheader("Age vs. Income")
        fig, ax = plt.subplots()
        sns.scatterplot(data=filtered_df.sample(2000), x='AGE_YEARS', y='AMT_INCOME_TOTAL', hue='TARGET', palette=['#66b3ff','#ff9999'], alpha=0.5, ax=ax)
        ax.set_xlabel("Age (Years)")
        ax.set_ylabel("Total Income")
        ax.set_yscale('log')
        st.pyplot(fig)

    # --- Row 3: Employment vs. Credit & Education vs. Credit ---
    graph_col5, graph_col6 = st.columns(2)
    with graph_col5:
        st.subheader("Employment Years vs. Credit Amount")
        fig, ax = plt.subplots()
        sns.scatterplot(data=filtered_df.sample(2000), x='EMPLOYMENT_YEARS', y='AMT_CREDIT', hue='TARGET', palette=['#66b3ff','#ff9999'], alpha=0.5, ax=ax)
        ax.set_xlabel("Employment (Years)")
        ax.set_ylabel("Credit Amount")
        st.pyplot(fig)

    with graph_col6:
        st.subheader("Credit Amount by Education")
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered_df, y='NAME_EDUCATION_TYPE', x='AMT_CREDIT', hue='TARGET', palette=['#66b3ff','#ff9999'], ax=ax)
        ax.set_xlabel("Credit Amount")
        ax.set_ylabel("Education Type")
        ax.set_xscale('log')
        st.pyplot(fig)

    # --- Row 4: Filter-Responsive Bar Charts ---
    st.markdown("---")
    st.header("Interactive Slice-and-Dice")
    st.write("These charts respond to the global filters in the sidebar.")
    graph_col7, graph_col8 = st.columns(2)
    with graph_col7:
        st.subheader("Default Rate by Gender (Filtered)")
        gender_default_rate = filtered_df.groupby('CODE_GENDER')['TARGET'].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(x='CODE_GENDER', y='TARGET', data=gender_default_rate, ax=ax, palette='viridis')
        ax.set_ylabel("Default Rate")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        st.pyplot(fig)

    with graph_col8:
        st.subheader("Default Rate by Housing Type (Filtered)")
        housing_default_rate = filtered_df.groupby('NAME_HOUSING_TYPE')['TARGET'].mean().sort_values(ascending=False).reset_index()
        fig, ax = plt.subplots()
        sns.barplot(y='NAME_HOUSING_TYPE', x='TARGET', data=housing_default_rate, ax=ax, palette='plasma')
        ax.set_xlabel("Default Rate")
        ax.set_ylabel("Housing Type")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        st.pyplot(fig)
        
    # --- Row 5: Pair Plot ---
    st.subheader("Pair Plot of Key Financial Variables")
    st.write("Exploring relationships between variables. (Using a small sample for speed)")
    pairplot_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AGE_YEARS', 'TARGET']
    pairplot_sample = filtered_df[pairplot_cols].sample(500)
    fig = sns.pairplot(pairplot_sample, hue='TARGET', palette=['#66b3ff','#ff9999'])
    st.pyplot(fig)

    # --- Narrative Insights ---
    st.header("Insights & Candidate Policy Rules")
    st.markdown("""
    * **Strongest Drivers:** The correlation analysis confirms our findings from previous pages. Key factors negatively correlated with default (meaning they are signs of a GOOD applicant) include longer employment, older age, and higher income. Factors positively correlated with default (warning signs) include higher DTI and LTI ratios.
    * **Interactive Exploration:** The scatter plots allow for dynamic exploration. For example, by filtering for 'Low Income' applicants in the sidebar, the 'Age vs. Credit' plot reveals that younger, low-income applicants who take on high credit amounts are a significant risk segment, as indicated by the concentration of red dots (defaults).
    * **Candidate Policy Rules:**
        * **LTI Caps:** Based on the strong positive correlation between LTI and default, the bank could implement a policy to cap the LTI ratio, perhaps with stricter caps for younger applicants or those with shorter employment histories.
        * **Income Verification:** For applicants in certain high-risk occupations (identified on Page 3) or with low education levels, requiring additional income verification or a lower initial credit limit could mitigate risk.
        * **Age & Employment Tiers:** Consider creating risk tiers. Applicants under a certain age (e.g., 25) and with less than 2 years of employment might automatically be placed in a higher-risk category requiring more scrutiny.

    """)




