import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def load_data(path="../csv/Mall_Customers.csv"):
    """Loads the dataset, handles nulls, duplicates, and removes ID columns."""
    df = pd.read_csv(path, na_values=['null'])
    df.replace("null", np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    if df.columns[0].lower().endswith('id'):
        df.drop(columns=[df.columns[0]], inplace=True)
    
    return df

def data_summary(df):
    numeric_df = df.select_dtypes(include=['number'])
    
    return {
        "head": df.head().to_dict(orient="records"),  # list of dicts
        "shape": [int(x) for x in df.shape],          # convert numpy int to int
        "describe": df.describe().to_dict(),          # convert DataFrame to dict
        "corr": numeric_df.corr().to_dict(),          # convert DataFrame to dict
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum())
    }



def clean_data(df):
    """Remove outliers and reset index"""

    df = df.drop_duplicates()

    for col in df.select_dtypes(include='object'):
        df[col] = pd.to_numeric(df[col], errors='ignore')

    numeric_df = df.select_dtypes(include=['number'])
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1

    outliers_iqr = df[((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]
    # print(f"Number of outlier rows: {outliers_iqr.shape[0]}")
    # print(outliers_iqr.head())  

    df_clean = df[~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]
    # numeric_df1 = df_clean.select_dtypes(include=['number'])
    # print(f"Removed rows: {df.shape[0] - df_clean.shape[0]}")
    # print(f"Remaining rows: {df_clean.shape[0]}")

    df_clean.reset_index(drop=True, inplace=True)
    return df_clean



def scale(df_clean):
    numeric_col = df_clean.select_dtypes(include=['number']).columns
    X = df_clean[numeric_col].values

    # 1️⃣ Power transform then scale
    pt = PowerTransformer(method='yeo-johnson')
    X_pt = pt.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pt)

    # 2️⃣ PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

    # 4️⃣ Elbow method to choose K
    inertia = []
    K = range(1,10)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    distances = [abs(K[i+1]-K[i]) + abs(inertia[i+1]-inertia[i]) for i in range(len(K)-1)]
    elbow_index = np.argmin(distances) + 1 


    # 5️⃣ Fit KMeans with chosen k (e.g., k=3)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df_clean['clusters'] = clusters
    df_pca['clusters'] = clusters

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_variance_explained": float(sum(pca.explained_variance_ratio_)),
        "cluster_centers": kmeans.cluster_centers_.tolist(),
        "df_clean_head": df_clean.head(20).to_dict(orient="records"),
        "df_pca_head": df_pca.head(20).to_dict(orient="records")
    }

