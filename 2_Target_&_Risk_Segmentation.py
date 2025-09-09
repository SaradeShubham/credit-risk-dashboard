# ===PAGE NO 2 ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, create_sidebar_filters

st.set_page_config(layout="wide")

# === Load and Filter Data ==
df = load_data()
filtered_df = create_sidebar_filters(df)

# == Page Title and Introduction ==
st.title("🎯 Page 2: Target & Risk Segmentation")
st.write(f"Displaying data for **{len(filtered_df):,}** applicants based on filters.")

# === KPIs for Risk Segmentation ===
st.header("Key Risk Indicators")

# Creating dataframes for defaulters and non-defaulters for easy comparison ===
defaulters_df = filtered_df[filtered_df['TARGET'] == 1]
non_defaulters_df = filtered_df[filtered_df['TARGET'] == 0]

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric(label="Applicants in Selection 👥", value=f"{len(filtered_df):,}")
kpi2.metric(label="Total Defaults 📉", value=f"{len(defaulters_df):,}")
kpi3.metric(label="Overall Default Rate %", value=f"{filtered_df['TARGET'].mean():.2%}")
kpi4.metric(label="Avg Age (Defaulters)", value=f"{defaulters_df['AGE_YEARS'].mean():.1f}")
kpi5.metric(label="Avg Employment Yrs (Defaulters)", value=f"{defaulters_df['EMPLOYMENT_YEARS'].mean():.1f}")

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

# == Graphs for Risk Segmentation ==
st.header("Visual Risk Segmentation")

graph_col1, graph_col2 = st.columns(2)
with graph_col1:
    st.subheader("Default vs. Repaid Counts")
    fig, ax = plt.subplots()
    sns.countplot(x='TARGET', data=filtered_df, ax=ax, palette=['#66b3ff','#ff9999'], hue='TARGET', legend=False)
    ax.set_xticklabels(['Repaid (0)', 'Default (1)'])
    ax.set_xlabel("Loan Status")
    st.pyplot(fig)
    plt.close(fig) 

with graph_col2:
    st.subheader("Default Rate by Gender")
    gender_default_rate = filtered_df.groupby('CODE_GENDER')['TARGET'].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x='CODE_GENDER', y='TARGET', data=gender_default_rate, ax=ax, palette='viridis', hue='CODE_GENDER', legend=False)
    ax.set_ylabel("Default Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    st.pyplot(fig)
    plt.close(fig) 

graph_col3, graph_col4 = st.columns(2)
with graph_col3:
    st.subheader("Default Rate by Education")
    edu_default_rate = filtered_df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots()
    sns.barplot(y='NAME_EDUCATION_TYPE', x='TARGET', data=edu_default_rate, ax=ax, palette='plasma', hue='NAME_EDUCATION_TYPE', legend=False)
    ax.set_xlabel("Default Rate")
    ax.set_ylabel("Education Type")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    st.pyplot(fig)
    plt.close(fig) 

with graph_col4:
    st.subheader("Default Rate by Housing Type")
    housing_default_rate = filtered_df.groupby('NAME_HOUSING_TYPE')['TARGET'].mean().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots()
    sns.barplot(y='NAME_HOUSING_TYPE', x='TARGET', data=housing_default_rate, ax=ax, palette='magma', hue='NAME_HOUSING_TYPE', legend=False)
    ax.set_xlabel("Default Rate")
    ax.set_ylabel("Housing Type")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    st.pyplot(fig)
    plt.close(fig) 

graph_col5, graph_col6 = st.columns(2)
with graph_col5:
    st.subheader("Income Distribution by Loan Status")
    fig, ax = plt.subplots()
    sns.boxplot(x='TARGET', y='AMT_INCOME_TOTAL', data=filtered_df, ax=ax, palette=['#66b3ff','#ff9999'])
    ax.set_xticklabels(['Repaid (0)', 'Default (1)'])
    ax.set_yscale('log')
    st.pyplot(fig)
    plt.close(fig) 

with graph_col6:
    st.subheader("Credit Amount by Loan Status")
    fig, ax = plt.subplots()
    sns.boxplot(x='TARGET', y='AMT_CREDIT', data=filtered_df, ax=ax, palette=['#66b3ff','#ff9999'])
    ax.set_xticklabels(['Repaid (0)', 'Default (1)'])
    ax.set_yscale('log')
    st.pyplot(fig)
    plt.close(fig) 

graph_col7, graph_col8 = st.columns(2)
with graph_col7:
    st.subheader("Age Distribution by Loan Status")
    fig, ax = plt.subplots()
    sns.violinplot(x='TARGET', y='AGE_YEARS', data=filtered_df, ax=ax, palette=['#66b3ff','#ff9999'])
    ax.set_xticklabels(['Repaid (0)', 'Default (1)'])
    st.pyplot(fig)
    plt.close(fig)

with graph_col8:
    st.subheader("Employment Years by Loan Status")
    fig, ax = plt.subplots()
    sns.histplot(data=filtered_df, x='EMPLOYMENT_YEARS', hue='TARGET', multiple='stack', ax=ax, palette=['#66b3ff','#ff9999'], bins=30)
    st.pyplot(fig)
    plt.close(fig) 

graph_col9, graph_col10 = st.columns(2)
with graph_col9:
    st.subheader("Contract Type vs Loan Status")
    contract_status = pd.crosstab(filtered_df['NAME_CONTRACT_TYPE'], filtered_df['TARGET'])
    fig, ax = plt.subplots()
    contract_status.plot(kind='bar', stacked=True, ax=ax, color=['#66b3ff','#ff9999'])
    ax.set_ylabel("Number of Applicants")
    ax.set_xlabel("Contract Type")
    ax.tick_params(axis='x', rotation=0)
    st.pyplot(fig)
    plt.close(fig) 

with graph_col10:
    st.subheader("Default Rate by Family Status")
    family_default_rate = filtered_df.groupby('NAME_FAMILY_STATUS')['TARGET'].mean().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots()
    sns.barplot(y='NAME_FAMILY_STATUS', x='TARGET', data=family_default_rate, ax=ax, palette='cividis', hue='NAME_FAMILY_STATUS', legend=False)
    ax.set_xlabel("Default Rate")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    st.pyplot(fig)
    plt.close(fig) 

# == Narrative Insights ==
st.header("Insights on Risk Segmentation")
st.markdown("""
- **Highest Risk Segments:** The bar charts consistently highlight specific groups with higher-than-average default rates. For example, applicants who are single, have 'Secondary' education, and live in 'Rented apartments' or 'With parents' tend to show a higher propensity to default.
- **Lowest Risk Segments:** Conversely, applicants who are 'Married', have 'Higher education', and are 'House / apartment' owners exhibit the lowest default rates. This suggests financial stability and life stage are strong indicators of repayment ability.
- **Financial Indicators:** The KPIs and boxplots show that defaulters, on average, have slightly lower incomes and request slightly higher credit amounts compared to those who repay. The violin plot for age shows that younger applicants have a higher concentration in the defaulter group.
""")