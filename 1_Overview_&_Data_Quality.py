# ====PAGE NO 1 ====
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, create_sidebar_filters

st.set_page_config(layout="wide")

# == Load and Filter Data ==
df = load_data()
filtered_df = create_sidebar_filters(df)

# == Page Title and Intro ==
st.title("📊 Page 1: Overview & Data Quality")
st.write(f"Displaying data for **{len(filtered_df):,}** applicants based on filters.")

# == KPIs for Overview ==
st.header("Key Performance Indicators")

kpi_cols = st.columns(5)
kpi_cols[0].metric(label="Total Applicants 👥", value=f"{filtered_df['SK_ID_CURR'].nunique():,}")
kpi_cols[1].metric(label="Default Rate 👎", value=f"{filtered_df['TARGET'].mean():.2%}")
kpi_cols[2].metric(label="Repaid Rate 👍", value=f"{1 - filtered_df['TARGET'].mean():.2%}")
kpi_cols[3].metric(label="Median Applicant Age", value=f"{filtered_df['AGE_YEARS'].median():.1f} Yrs")
kpi_cols[4].metric(label="Avg Credit Amount 💳", value=f"₹{filtered_df['AMT_CREDIT'].mean():,.0f}")

kpi_cols_2 = st.columns(5)
kpi_cols_2[0].metric(label="Total Features 📊", value=filtered_df.shape[1])
kpi_cols_2[1].metric(label="Numerical Features", value=filtered_df.select_dtypes(include=np.number).shape[1])
kpi_cols_2[2].metric(label="Categorical Features", value=filtered_df.select_dtypes(include=['object', 'category']).shape[1])
kpi_cols_2[3].metric(label="Median Annual Income 💵", value=f"₹{filtered_df['AMT_INCOME_TOTAL'].median():,.0f}")
kpi_cols_2[4].metric(label="Avg Employment Years", value=f"{filtered_df['EMPLOYMENT_YEARS'].mean():.1f} Yrs")

# == Graphs for Overview ==
st.header("Visualizations of Key Features")

graph_col1, graph_col2 = st.columns(2)
with graph_col1:
    st.subheader("Loan Status Distribution")
    fig, ax = plt.subplots()
    filtered_df['TARGET'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=['#66b3ff','#ff9999'], labels=['Repaid', 'Default'])
    ax.set_ylabel('')
    st.pyplot(fig)
    plt.close(fig) # FIX: Close the figure to free memory

with graph_col2:
    st.subheader("Gender Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='CODE_GENDER', data=filtered_df, ax=ax, palette='viridis', hue='CODE_GENDER', legend=False)
    ax.set_xlabel("Gender")
    ax.set_ylabel("Number of Applicants")
    st.pyplot(fig)
    plt.close(fig) 

graph_col3, graph_col4 = st.columns(2)
with graph_col3:
    st.subheader("Total Income Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['AMT_INCOME_TOTAL'], bins=30, kde=True, ax=ax, color='salmon')
    ax.set_xlabel("Total Income")
    st.pyplot(fig)
    plt.close(fig) 

with graph_col4:
    st.subheader("Credit Amount Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['AMT_CREDIT'], bins=30, kde=True, ax=ax, color='lightgreen')
    ax.set_xlabel("Credit Amount")
    st.pyplot(fig)
    plt.close(fig) 

graph_col5, graph_col6 = st.columns(2)
with graph_col5:
    st.subheader("Total Income Boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(y='AMT_INCOME_TOTAL', data=filtered_df, ax=ax, color='salmon')
    st.pyplot(fig)
    plt.close(fig) 

with graph_col6:
    st.subheader("Credit Amount Boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(y='AMT_CREDIT', data=filtered_df, ax=ax, color='lightgreen')
    st.pyplot(fig)
    plt.close(fig) 

graph_col7, graph_col8 = st.columns(2)
with graph_col7:
    st.subheader("Family Status")
    fig, ax = plt.subplots()
    sns.countplot(y='NAME_FAMILY_STATUS', data=filtered_df, ax=ax, palette='plasma', hue='NAME_FAMILY_STATUS', legend=False)
    ax.set_xlabel("Number of Applicants")
    ax.set_ylabel("Family Status")
    st.pyplot(fig)
    plt.close(fig) 

with graph_col8:
    st.subheader("Education Type")
    fig, ax = plt.subplots()
    sns.countplot(y='NAME_EDUCATION_TYPE', data=filtered_df, ax=ax, palette='magma', hue='NAME_EDUCATION_TYPE', legend=False)
    ax.set_xlabel("Number of Applicants")
    ax.set_ylabel("Education Type")
    st.pyplot(fig)
    plt.close(fig) 

graph_col9, graph_col10 = st.columns(2)
with graph_col9:
    st.subheader("Housing Type")
    fig, ax = plt.subplots()
    sns.countplot(y='NAME_HOUSING_TYPE', data=filtered_df, ax=ax, palette='cividis', hue='NAME_HOUSING_TYPE', legend=False)
    ax.set_xlabel("Number of Applicants")
    ax.set_ylabel("Housing Type")
    st.pyplot(fig)
    plt.close(fig) 

with graph_col10:
    st.subheader("Applicant Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['AGE_YEARS'], bins=30, kde=True, ax=ax, color='skyblue')
    ax.set_xlabel("Age (Years)")
    st.pyplot(fig)
    plt.close(fig) 

# == Narrative Insights ==
st.header("Insights")
st.markdown("""
- **Portfolio Risk:** The overall default rate serves as a critical baseline for our analysis. This is the average risk across the entire portfolio selected by the filters.
- **Applicant Demographics:** The applicant base is heavily skewed towards a younger demographic. There is a significant majority of female applicants.
- **Financial Profile:** The histograms for income and credit amount show a strong right-skew, indicating that the vast majority of applicants have lower incomes and apply for smaller loan amounts.
""")
