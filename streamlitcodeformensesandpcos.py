import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Streamlit App Configuration
st.set_page_config(page_title="Menstrual Health and PCOD Risk Dashboard", layout="wide")

st.title("🩸 Menstrual Health and PCOD Risk Analysis")
st.write("Upload your dataset below to explore menstrual health patterns and possible PCOD risk factors.")

# File Upload
uploaded_file = st.file_uploader("📂 Upload your CSV file (e.g., periods.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully ✅")

    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    st.write(f"Shape: {df.shape}")
    st.write(f"Duplicate Rows: {df.duplicated().sum()}")

    df = df.drop_duplicates()
    st.write("✅ Duplicates removed!")

    st.write("### Missing Values")
    st.write(df.isnull().sum())

    st.write("### Data Info")
    buffer = []
    df.info(buf=buffer)
    info_str = "\n".join(buffer)
    st.text(info_str)

    st.write("### Descriptive Statistics")
    st.dataframe(df.describe())

    # Data Cleaning
    if "Height" in df.columns:
        df['Height'] = df['Height'].astype(str).str.strip()
        df['Height'] = df['Height'].str.replace(' ', "'")
        df['Height'] = df['Height'].str.replace('"', '')

        def convert_to_inches(x):
            try:
                if "'" in x:
                    feet, inches = map(int, x.split("'"))
                    return feet * 12 + inches
                else:
                    return pd.to_numeric(x, errors='coerce')
            except ValueError:
                return pd.to_numeric(x, errors='coerce')

        df['Height'] = df['Height'].apply(convert_to_inches)

    if "Unusual_Bleeding" in df.columns:
        df['Unusual_Bleeding'] = df['Unusual_Bleeding'].astype(str).str.lower().map({'yes': 'Yes', 'no': 'No'})

    if "Income" in df.columns:
        df['Income'] = df['Income'].apply(lambda x: 'Low' if x == 0 else 'High')

    if "Menses_score" in df.columns:
        df['Menses_score'] = df['Menses_score'].apply(lambda x: 'Low' if x <= 3 else 'High')

    # Feature Classification
    def classify_features(df):
        categorical_features, discrete_features, continuous_features = [], [], []
        for column in df.columns:
            if df[column].dtype == 'object':
                categorical_features.append(column)
            elif df[column].dtype in ['int64', 'float64']:
                if df[column].nunique() < 10:
                    discrete_features.append(column)
                else:
                    continuous_features.append(column)
        return categorical_features, discrete_features, continuous_features

    categorical, discrete, continuous = classify_features(df)

    st.subheader("Feature Classification")
    st.write("Categorical:", categorical)
    st.write("Discrete:", discrete)
    st.write("Continuous:", continuous)

    # Visualizations
    st.subheader("📊 Visualizations")

    # Sidebar options
    plot_type = st.sidebar.selectbox("Select Plot Type", ["Count Plot", "Pie Chart", "Histogram", "Box Plot", "Correlation Heatmap"])

    if plot_type == "Count Plot":
        col = st.selectbox("Select Categorical Column", categorical)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x=col, data=df, palette="hls")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif plot_type == "Pie Chart":
        col = st.selectbox("Select Column for Pie Chart", categorical + discrete)
        fig, ax = plt.subplots(figsize=(6, 6))
        df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
        plt.ylabel('')
        st.pyplot(fig)

    elif plot_type == "Histogram":
        col = st.selectbox("Select Continuous Column", continuous)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df[col], kde=True, bins=20, color='teal')
        st.pyplot(fig)

    elif plot_type == "Box Plot":
        cat_col = st.selectbox("Select Categorical Column", categorical)
        cont_col = st.selectbox("Select Continuous Column", continuous)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x=cat_col, y=cont_col, data=df, palette="coolwarm")
        st.pyplot(fig)

    elif plot_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[continuous].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        st.pyplot(fig)

    st.success("Analysis Completed ✅")

else:
    st.info("Please upload a CSV file to start the analysis.")
