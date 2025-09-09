# Home.py === Home Page
import streamlit as st

st.set_page_config(
    page_title="Home Credit Risk Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Home Credit Default Risk Dashboard")

st.markdown("""
Welcome to the Home Credit Default Risk analysis dashboard. This project provides a comprehensive exploration of a loan application dataset to uncover key insights related to credit risk.

The dashboard has been built following a professional data science workflow:
- **Offline Pre-processing:** A large, raw CSV file was cleaned, transformed, and saved into an efficient Parquet format.
- **Interactive Visualization:** The app uses the clean data to generate interactive charts and KPIs.
- **Multi-Page Structure:** The analysis is organized into logical sections for clarity.

**👈 Please select a page from the sidebar** to begin your analysis.

### Dashboard Pages:
- **Overview & Data Quality:** A high-level summary of the applicant portfolio.
- **Target & Risk Segmentation:** Analysis of default rates across different customer segments.
- **Demographics & Household:** A deep dive into the demographic profile of the applicants.
- **Financial Health:** Exploration of applicants' financial stability and affordability.
- **Correlations & Drivers:** Identifying the key factors that correlate with loan defaults.
""")