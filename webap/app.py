import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_URL = "https://kailo.onrender.com"

st.set_page_config(page_title="DataReveal Dashboard", layout="wide")

st.title("📊 DataReveal Dashboard")
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Preview Data", "Clean Data", "Scaling & Clustering", "EDA"])

# -----------------------------------
if page == "Home":
    st.subheader("Home")
    st.write("Welcome to DataReveal! Fetching summary from FastAPI...")
    st.title("DataReveal: Upload Your CSV")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
            files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
            resp = requests.post(f"{BASE_URL}/upload_csv", files=files)

            if resp.ok:
                st.success("File uploaded successfully!")
                data_info = resp.json()
                st.write("Shape:", data_info["shape"])
                st.write("Columns:", data_info["columns"])

                summary_resp = requests.get(f"{BASE_URL}/")
                if resp.ok:
                    data = resp.json()
                    st.write("Columns:", data["columns"])
                    st.write("Shape:", data["shape"])
                    st.write("Null counts:", data["null_counts"])
                    st.write("Duplicate rows:", data["duplicates"])
                    st.write("Data")
                    st.dataframe(pd.DataFrame(data["head"]))
                else:
                    st.error("Failed to fetch data summary.")
            else:
                st.error("Failed to upload file.")
    

# -----------------------------------
elif page == "Preview Data":
    st.subheader("Preview Dataset")
    limit = st.slider("Number of rows to preview", 1, 50, 5)
    resp = requests.get(f"{BASE_URL}/preview?limit={limit}")
    if resp.ok:
        df_preview = pd.DataFrame(resp.json())
        st.dataframe(df_preview)
    else:
        st.error("Failed to fetch preview data.")

# -----------------------------------
elif page == "Clean Data":
    st.subheader("Clean Data")
    if st.button("Clean Data"):
        resp = requests.get(f"{BASE_URL}/clean")
        if resp.ok:
            data = resp.json()
            st.success("Data cleaned successfully!")

            # -------------------------------
            st.subheader("Summary before cleaning")
            for key, value in data["summary_before_cleaning"].items():
                st.write(f"**{key}**")
                if isinstance(value, list): 
                    st.write(value)
                elif isinstance(value, dict):
                    st.dataframe(pd.DataFrame.from_dict(value, orient='index'))
                else:
                    st.write(value)

            st.subheader("Summary after cleaning")
            for key, value in data["summary_after_cleaning"].items():
                st.write(f"**{key}**")
                if isinstance(value, list):
                    st.write(value)
                elif isinstance(value, dict):
                    st.dataframe(pd.DataFrame.from_dict(value, orient='index'))
                else:
                    st.write(value)

            df_clean_head = pd.DataFrame(data["df_clean"])
            numeric_cols = df_clean_head.select_dtypes(include=['number'])

            st.subheader("📦 Boxplots for Numeric Features after cleaning")

            for col in numeric_cols.columns:
                st.write(f"Boxplot of {col}")
                fig, ax = plt.subplots(figsize=(4,4))
                sns.boxplot(y=df_clean_head[col], ax=ax)
                st.pyplot(fig)



        else:
            st.error("Failed to clean data.")

# -----------------------------------
elif page == "EDA":
    st.header("Exploratory Data Analysis")

    resp = requests.get(f"{BASE_URL}/preview?limit=200")  
    if resp.ok:
        df_eda = pd.DataFrame(resp.json())
        st.write("Dataset shape:", df_eda.shape)
        st.dataframe(df_eda.head())

        if st.button("Run EDA"):
            st.subheader("✅ Column Types & Null Values")
            st.write(df_eda.dtypes)
            st.write("Null Values per Column:")
            st.write(df_eda.isnull().sum())

            # ---------------------------
            st.subheader("🔢 Histograms for Numeric Features")
            numeric_cols = df_eda.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                st.write(f"Histogram of {col}")
                st.bar_chart(df_eda[col])

            # ---------------------------
            st.subheader("📦 Boxplots for Numeric Features")
            for col in numeric_cols:
                st.write(f"Boxplot of {col}")
                fig, ax = plt.subplots()
                sns.boxplot(y=df_eda[col], ax=ax)
                st.pyplot(fig)

            # ---------------------------
            st.subheader("🔗 Correlation Heatmap")
            corr = df_eda[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            # ---------------------------
            st.subheader("📊 Pairplot")
            fig = sns.pairplot(df_eda[numeric_cols])
            st.pyplot(fig)

            # ---------------------------
            st.subheader("Categorical Columns Analysis")
            cat_cols = df_eda.select_dtypes(include=['object']).columns
            for col in cat_cols:
                st.write(f"Value counts for {col}")
                st.bar_chart(df_eda[col].value_counts())
    else:
        st.error("Failed to fetch dataset for EDA.")

# -----------------------------------
elif page == "Scaling & Clustering":
    st.subheader("Scaling, PCA, and KMeans Clustering")
    if st.button("Run Scaling & Clustering"):
        resp = requests.get(f"{BASE_URL}/scale")
        if resp.ok:
            data = resp.json()
            
            st.write("Explained Variance Ratio:", data["explained_variance_ratio"])
            st.write("Total Variance Explained:", data["total_variance_explained"])
            st.write("Cluster Centers:")
            st.json(data["cluster_centers"])

            df_clean = pd.DataFrame(data["df_clean_head"])
            df_pca = pd.DataFrame(data["df_pca_head"])

            st.write("PCA Scatter Plot:")
            fig = px.scatter(df_pca, x="PC1", y="PC2", color="clusters",
                             title="PCA 2D Projection with Clusters")
            st.plotly_chart(fig, use_container_width=True)

            numeric_cols = df_clean.select_dtypes(include=['number']).columns
            st.subheader("📦 Boxplots for Numeric Features after Scaling & Clustering")
            for col in numeric_cols:
                st.write(f"Boxplot of {col}")
                fig, ax = plt.subplots(figsize=(8,4))
                sns.boxplot(y=df_clean[col], ax=ax)
                st.pyplot(fig)

        else:
            st.error("Failed to scale and cluster data.")
