#!/usr/bin/env python
# coding: utf-8

# In[2]:


from preprocess import df_clean


# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[22]:


df_clean.head()


# In[23]:


df = df_clean


# In[24]:


df_clean.info()


# In[35]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# In[78]:


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

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance explained by 2 components:", sum(pca.explained_variance_ratio_))

    # 3️⃣ Plot histograms after transformation
    for i, col in enumerate(numeric_col):
        plt.figure(figsize=(10,5))
        sns.histplot(X_scaled[:, i], kde=True)
        plt.title(f'Distribution of {col} after Transformation')
        plt.show()

    # 4️⃣ Elbow method to choose K
    inertia = []
    K = range(1,10)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    distances = [abs(K[i+1]-K[i]) + abs(inertia[i+1]-inertia[i]) for i in range(len(K)-1)]
    elbow_index = np.argmin(distances) + 1 


    plt.figure(figsize=(8,5))
    plt.plot(K, inertia, 'o-', color='blue')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()

    # 5️⃣ Fit KMeans with chosen k (e.g., k=3)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df_clean['clusters'] = clusters
    df_pca['clusters'] = clusters

    print("Cluster centers (scaled space):\n", kmeans.cluster_centers_)

    # 6️⃣ Plot clusters on PCA
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='PC1', y='PC2', hue='clusters', data=df_pca, palette='Set2', s=100)
    plt.title('K-Means Clustering on PCA Components')
    plt.show()

    # 7️⃣ Pairplot (all numeric columns)
    df_plot = df_clean.copy()
    df_plot['clusters'] = df_plot['clusters'].astype(str)
    sns.pairplot(df_plot, vars=numeric_col, hue='clusters', palette='Set2', diag_kind='kde')
    plt.suptitle('Pairplot of Clusters', y=1.02)
    plt.show()

    return df_clean, df_pca


# In[79]:


scale(df_clean)


# In[83]:


df_clean.head()


# In[84]:


df_clean.columns


# In[ ]:




