# ===PAGE NO 4====
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
st.title("💰 Page 4: Financial Health & Affordability")
st.write(f"Displaying data for **{len(filtered_df):,}** applicants based on filters.")

# == KPIs for Financial Health ==
st.header("Key Financial Indicators")

# Creating dataframes for defaulters and non-defaulters==
defaulters_df = filtered_df[filtered_df['TARGET'] == 1]
non_defaulters_df = filtered_df[filtered_df['TARGET'] == 0]

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric(label="Avg Annual Income 💵", value=f"₹{filtered_df['AMT_INCOME_TOTAL'].mean():,.0f}")
kpi2.metric(label="Median Annual Income", value=f"₹{filtered_df['AMT_INCOME_TOTAL'].median():,.0f}")
kpi3.metric(label="Avg Credit Amount 💳", value=f"₹{filtered_df['AMT_CREDIT'].mean():,.0f}")
kpi4.metric(label="Avg Loan Annuity", value=f"₹{filtered_df['AMT_ANNUITY'].mean():,.0f}")
kpi5.metric(label="Avg Goods Price", value=f"₹{filtered_df['AMT_GOODS_PRICE'].mean():,.0f}")

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

# === Graphs for Financial Health ==
st.header("Visual Financial Analysis")

graph_col1, graph_col2 = st.columns(2)
with graph_col1:
    st.subheader("Income Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['AMT_INCOME_TOTAL'], kde=True, ax=ax, color='blue', bins=30)
    st.pyplot(fig)
    plt.close(fig) # FIX

with graph_col2:
    st.subheader("Credit Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['AMT_CREDIT'], kde=True, ax=ax, color='green', bins=30)
    st.pyplot(fig)
    plt.close(fig) # FIX

graph_col3, graph_col4 = st.columns(2)
with graph_col3:
    st.subheader("Income vs. Credit Amount")
    fig, ax = plt.subplots()
    sns.scatterplot(x='AMT_INCOME_TOTAL', y='AMT_CREDIT', data=filtered_df.sample(2000), 
                    ax=ax, alpha=0.3, hue='TARGET', palette=['#66b3ff','#ff9999'])
    ax.ticklabel_format(style='plain', axis='both')
    st.pyplot(fig)
    plt.close(fig) # FIX

with graph_col4:
    st.subheader("Income vs. Annuity")
    fig, ax = plt.subplots()
    sns.scatterplot(x='AMT_INCOME_TOTAL', y='AMT_ANNUITY', data=filtered_df.sample(2000), 
                    ax=ax, alpha=0.3, hue='TARGET', palette=['#66b3ff','#ff9999'])
    ax.ticklabel_format(style='plain', axis='both')
    st.pyplot(fig)
    plt.close(fig) # FIX

graph_col5, graph_col6 = st.columns(2)
with graph_col5:
    st.subheader("Income by Loan Status")
    fig, ax = plt.subplots()
    sns.boxplot(x='TARGET', y='AMT_INCOME_TOTAL', data=filtered_df, ax=ax, palette=['#66b3ff','#ff9999'])
    ax.set_yscale('log')
    ax.set_xticklabels(['Repaid (0)', 'Default (1)'])
    st.pyplot(fig)
    plt.close(fig) # FIX

with graph_col6:
    st.subheader("Credit by Loan Status")
    fig, ax = plt.subplots()
    sns.boxplot(x='TARGET', y='AMT_CREDIT', data=filtered_df, ax=ax, palette=['#66b3ff','#ff9999'])
    ax.set_yscale('log')
    ax.set_xticklabels(['Repaid (0)', 'Default (1)'])
    st.pyplot(fig)
    plt.close(fig) # FIX

graph_col7, graph_col8 = st.columns(2)
with graph_col7:
    st.subheader("Annuity Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['AMT_ANNUITY'], kde=True, ax=ax, color='red', bins=30)
    st.pyplot(fig)
    plt.close(fig) # FIX

with graph_col8:
    st.subheader("Default Rate by Income Bracket")
    income_default_rate = filtered_df.groupby('INCOME_BRACKET')['TARGET'].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x='INCOME_BRACKET', y='TARGET', data=income_default_rate, ax=ax, palette='coolwarm', hue='INCOME_BRACKET', legend=False)
    ax.set_ylabel("Default Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    st.pyplot(fig)
    plt.close(fig) # FIX

graph_col9, graph_col10 = st.columns(2)
with graph_col9:
    st.subheader("Joint Income-Credit Density")
    fig, ax = plt.subplots()
    sns.kdeplot(data=filtered_df.sample(2000), x="AMT_INCOME_TOTAL", y="AMT_CREDIT", ax=ax, fill=True, thresh=0.05, cmap="viridis")
    st.pyplot(fig)
    plt.close(fig) # FIX

with graph_col10:
    st.subheader("Financial Variable Correlations")
    fig, ax = plt.subplots()
    corr_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DTI', 'LOAN_TO_INCOME', 'TARGET']
    corr_matrix = filtered_df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt='.2f', ax=ax, annot_kws={"size": 8})
    st.pyplot(fig)
    plt.close(fig) # FIX

# == Narrative Insights ==
st.header("Insights on Financial Health")
st.markdown("""
- **Affordability Thresholds:** The charts and KPIs reveal a strong link between debt ratios and default risk. Applicants with high Loan-to-Income (LTI) and Debt-to-Income (DTI) ratios are more likely to default, suggesting they may be over-leveraged.
- **Income is a Key Differentiator:** The 'Default Rate by Income Bracket' shows a clear, inverse relationship: as income increases, the default rate decreases. The KPIs also highlight a significant "Income Gap," showing that applicants who repay their loans have a meaningfully higher income, on average, than those who default.
- **Strong Correlations:** The final heatmap reveals expected strong positive correlations between credit amount, annuity, and the price of goods. More importantly, it shows a negative correlation between income and the key debt ratios, confirming that higher-income individuals tend to have more manageable debt.
""")