#+++PAGE NO 5======
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, create_sidebar_filters

st.set_page_config(layout="wide")

# ==Load and Filter Data==
df = load_data()
filtered_df = create_sidebar_filters(df)

# ==Page Title and Introduction===
st.title("🔍 Page 5: Correlations, Drivers & Interactive Slice-and-Dice")
st.write(f"Displaying data for **{len(filtered_df):,}** applicants based on filters.")

# == Correlation Calculation ==
numeric_cols = filtered_df.select_dtypes(include=np.number).columns
corr_matrix = filtered_df[numeric_cols].corr()
target_corr = corr_matrix['TARGET'].sort_values(ascending=False)

# == KPIs for Correlations====
st.header("Key Correlation Indicators")

kpi_col1, kpi_col2 = st.columns(2)
with kpi_col1:
    st.subheader("Top 5 Positive Correlations with Default")
    st.dataframe(target_corr.head(6)[1:], use_container_width=True)

with kpi_col2:
    st.subheader("Top 5 Negative Correlations with Default")
    st.dataframe(target_corr.tail(5), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

kpi3, kpi4, kpi5, kpi6, kpi7 = st.columns(5)
corr_age_target = corr_matrix.loc['AGE_YEARS', 'TARGET']
kpi3.metric(label="Corr(Age, TARGET)", value=f"{corr_age_target:.3f}")
corr_emp_target = corr_matrix.loc['EMPLOYMENT_YEARS', 'TARGET']
kpi4.metric(label="Corr(Employment, TARGET)", value=f"{corr_emp_target:.3f}")
corr_inc_credit = corr_matrix.loc['AMT_INCOME_TOTAL', 'AMT_CREDIT']
kpi5.metric(label="Corr(Income, Credit)", value=f"{corr_inc_credit:.3f}")
most_corr_income = corr_matrix['AMT_INCOME_TOTAL'].sort_values(ascending=False).index[1]
kpi6.metric(label="Most Correlated w/ Income", value=most_corr_income, help="Excluding itself")
strong_corrs = target_corr[abs(target_corr) > 0.05].shape[0] - 1
kpi7.metric(label="# Features w/ |corr| > 0.05", value=strong_corrs, help="Number of features with a notable correlation to default.")

# == Graphs for Correlations ==
st.header("Visual Correlation Analysis")

graph_col1, graph_col2 = st.columns([2, 1])
with graph_col1:
    st.subheader("Correlation Heatmap of Key Features")
    fig, ax = plt.subplots(figsize=(12, 10))
    heatmap_cols = ['TARGET', 'AGE_YEARS', 'EMPLOYMENT_YEARS', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DTI', 'LOAN_TO_INCOME']
    sns.heatmap(filtered_df[heatmap_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)
    plt.close(fig) # FIX

with graph_col2:
    st.subheader("Top Correlates with Default")
    top_corr = pd.concat([target_corr.head(6)[1:], target_corr.tail(5)]).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 10))
    sns.barplot(x=top_corr.values, y=top_corr.index, ax=ax, palette='vlag', hue=top_corr.index, legend=False)
    ax.set_xlabel("Correlation with TARGET")
    st.pyplot(fig)
    plt.close(fig) # FIX

graph_col3, graph_col4 = st.columns(2)
with graph_col3:
    st.subheader("Age vs. Credit Amount")
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df.sample(2000), x='AGE_YEARS', y='AMT_CREDIT', hue='TARGET', palette=['#66b3ff','#ff9999'], alpha=0.5, ax=ax)
    st.pyplot(fig)
    plt.close(fig) # FIX

with graph_col4:
    st.subheader("Age vs. Income")
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df.sample(2000), x='AGE_YEARS', y='AMT_INCOME_TOTAL', hue='TARGET', palette=['#66b3ff','#ff9999'], alpha=0.5, ax=ax)
    ax.set_yscale('log')
    st.pyplot(fig)
    plt.close(fig) # FIX

graph_col5, graph_col6 = st.columns(2)
with graph_col5:
    st.subheader("Employment Years vs. Credit Amount")
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df.sample(2000), x='EMPLOYMENT_YEARS', y='AMT_CREDIT', hue='TARGET', palette=['#66b3ff','#ff9999'], alpha=0.5, ax=ax)
    st.pyplot(fig)
    plt.close(fig) # FIX

with graph_col6:
    st.subheader("Credit Amount by Education")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, y='NAME_EDUCATION_TYPE', x='AMT_CREDIT', hue='TARGET', palette=['#66b3ff','#ff9999'], ax=ax)
    ax.set_xscale('log')
    st.pyplot(fig)
    plt.close(fig) # FIX
    
graph_col7, graph_col8 = st.columns(2)
with graph_col7:
    st.subheader("Income by Family Status")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, y='NAME_FAMILY_STATUS', x='AMT_INCOME_TOTAL', hue='TARGET', palette=['#66b3ff','#ff9999'], ax=ax)
    ax.set_xscale('log')
    st.pyplot(fig)
    plt.close(fig) # FIX

with graph_col8:
    st.subheader("Default Rate by Gender (Interactive)")
    gender_default_rate = filtered_df.groupby('CODE_GENDER')['TARGET'].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x='CODE_GENDER', y='TARGET', data=gender_default_rate, ax=ax, palette='viridis', hue='CODE_GENDER', legend=False)
    ax.set_ylabel("Default Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    st.pyplot(fig)
    plt.close(fig) # FIX

st.subheader("Pair Plot of Key Financial Variables")
st.write("Exploring relationships between variables. (Using a small sample for speed)")
pairplot_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AGE_YEARS', 'TARGET']
pairplot_sample = filtered_df[pairplot_cols].sample(500, random_state=1)
fig_pairplot = sns.pairplot(pairplot_sample, hue='TARGET', palette=['#66b3ff','#ff9999'])
st.pyplot(fig_pairplot)
plt.close(fig_pairplot.fig) # FIX

# == Narrative Insights ==
st.header("Insights & Candidate Policy Rules")
st.markdown("""
- **Strongest Drivers:** The correlation analysis confirms our findings from previous pages. Key factors negatively correlated with default (meaning they are signs of a GOOD applicant) include older age and longer employment. Factors positively correlated with default (warning signs) include higher DTI and LTI ratios.
- **Interactive Exploration:** The scatter plots allow for dynamic exploration. For example, by filtering for 'Low Income' applicants in the sidebar, the 'Age vs. Credit' plot reveals that younger, low-income applicants who take on high credit amounts are a significant risk segment.
- **Candidate Policy Rules:**
    - **LTI Caps:** Based on the strong positive correlation between LTI and default, the bank could implement a policy to cap the LTI ratio, perhaps with stricter caps for younger applicants.
    - **Income Verification:** For applicants in certain high-risk occupations or with low education levels, requiring additional income verification or a lower initial credit limit could mitigate risk.
""")