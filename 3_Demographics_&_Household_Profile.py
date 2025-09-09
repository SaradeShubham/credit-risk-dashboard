# PAGE NO 3
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

# == Page Title and Introduction ==
st.title("👨‍👩‍👧‍👦 Page 3: Demographics & Household Profile")
st.write(f"Displaying data for **{len(filtered_df):,}** applicants based on filters.")

# == KPIs for Demographics ==
st.header("Key Demographic Indicators")

# Creating dataframes for defaulters and non-defaulters ==
defaulters_df = filtered_df[filtered_df['TARGET'] == 1]
non_defaulters_df = filtered_df[filtered_df['TARGET'] == 0]

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
gender_dist = filtered_df['CODE_GENDER'].value_counts(normalize=True) * 100
kpi1.metric(label="% Female Applicants 👩", value=f"{gender_dist.get('F', 0):.1f}%")
kpi2.metric(label="% Male Applicants 👨", value=f"{gender_dist.get('M', 0):.1f}%")
kpi3.metric(label="Avg Age (Defaulters)", value=f"{defaulters_df['AGE_YEARS'].mean():.1f}")
kpi4.metric(label="Avg Age (Repaid)", value=f"{non_defaulters_df['AGE_YEARS'].mean():.1f}")
kpi5.metric(label="Avg Family Size 👨‍👩‍👧", value=f"{filtered_df['CNT_FAM_MEMBERS'].mean():.2f}")

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

# == Graphs for Demographics ==
st.header("Visual Demographic Profiles")

graph_col1, graph_col2 = st.columns(2)
with graph_col1:
    st.subheader("Age Distribution (All Applicants)")
    fig, ax = plt.subplots()
    sns.histplot(data=filtered_df, x='AGE_YEARS', kde=True, ax=ax, color='teal')
    st.pyplot(fig)
    plt.close(fig) 

with graph_col2:
    st.subheader("Age Distribution by Loan Status")
    fig, ax = plt.subplots()
    sns.histplot(data=filtered_df, x='AGE_YEARS', hue='TARGET', kde=True, ax=ax, palette=['#66b3ff','#ff9999'])
    st.pyplot(fig)
    plt.close(fig) 

graph_col3, graph_col4 = st.columns(2)
with graph_col3:
    st.subheader("Gender Distribution")
    fig, ax = plt.subplots()
    filtered_df['CODE_GENDER'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=['#ff9999','#66b3ff','grey'])
    ax.set_ylabel('')
    st.pyplot(fig)
    plt.close(fig) 

with graph_col4:
    st.subheader("Family Status Distribution")
    fig, ax = plt.subplots()
    sns.countplot(y='NAME_FAMILY_STATUS', data=filtered_df, ax=ax, palette='viridis', hue='NAME_FAMILY_STATUS', legend=False)
    ax.set_ylabel("Family Status")
    st.pyplot(fig)
    plt.close(fig) 

graph_col5, graph_col6 = st.columns(2)
with graph_col5:
    st.subheader("Education Distribution")
    fig, ax = plt.subplots()
    sns.countplot(y='NAME_EDUCATION_TYPE', data=filtered_df, ax=ax, palette='plasma', hue='NAME_EDUCATION_TYPE', legend=False)
    ax.set_ylabel("Education Type")
    st.pyplot(fig)
    plt.close(fig) 

with graph_col6:
    st.subheader("Top 10 Occupations")
    fig, ax = plt.subplots()
    top_10_occupations = filtered_df['OCCUPATION_TYPE'].value_counts().nlargest(10)
    sns.barplot(y=top_10_occupations.index, x=top_10_occupations.values, ax=ax, palette='cividis', hue=top_10_occupations.index, legend=False)
    st.pyplot(fig)
    plt.close(fig) 

graph_col7, graph_col8 = st.columns(2)
with graph_col7:
    st.subheader("Housing Type Distribution")
    fig, ax = plt.subplots()
    filtered_df['NAME_HOUSING_TYPE'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, wedgeprops=dict(width=0.4))
    ax.set_ylabel('')
    st.pyplot(fig)
    plt.close(fig) 

with graph_col8:
    st.subheader("Number of Children")
    fig, ax = plt.subplots()
    sns.countplot(x='CNT_CHILDREN', data=filtered_df, ax=ax, palette='magma', hue='CNT_CHILDREN', legend=False)
    ax.set_xlabel("Number of Children")
    st.pyplot(fig)
    plt.close(fig) 

graph_col9, graph_col10 = st.columns(2)
with graph_col9:
    st.subheader("Age vs. Loan Status")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, x='TARGET', y='AGE_YEARS', ax=ax, palette=['#66b3ff','#ff9999'])
    ax.set_xticklabels(['Repaid (0)', 'Default (1)'])
    st.pyplot(fig)
    plt.close(fig) 

with graph_col10:
    st.subheader("Demographic Correlation Heatmap")
    fig, ax = plt.subplots()
    corr_cols = ['AGE_YEARS', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'TARGET']
    corr_matrix = filtered_df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
    st.pyplot(fig)
    plt.close(fig) 

# == Narrative Insights ==
st.header("Insights on Demographics")
st.markdown("""
- **Life Stage Matters:** The boxplot and age distribution graphs show that defaulters tend to be slightly younger than those who repay. This, combined with the higher default rates for those living with parents, suggests younger applicants in earlier life stages represent a higher risk.
- **Family Structure & Stability:** Married applicants, who form the largest group, also have one of the lowest default rates. This indicates that family stability is correlated with financial reliability.
- **Weak Correlations:** The heatmap shows that individual demographic factors like age, number of children, and family size have very weak direct correlations with defaulting. This implies that risk is not driven by a single demographic factor but likely a complex interaction between demographics, financial health, and loan characteristics.
""")