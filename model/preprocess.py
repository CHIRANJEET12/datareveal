#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[94]:


from dataload import df


# In[95]:


from dataload import numeric_df


# In[96]:


df.head()


# In[97]:


def clean_data(df):
    print("🧹 Remove null rows/missing values")
    m = df.shape[0]
    df.dropna(inplace=True)
    if df.shape[0] == m:
        print("✅ No null or missing values")
    print(f"Shape after removing missing values: {df.shape}")

    df = df.drop_duplicates()
    print(f"Shape after removing duplicates: {df.shape}\n")

    print("🔢 Convert meaningful object columns to numeric")
    for col in df.select_dtypes(include='object'):
        df[col] = pd.to_numeric(df[col], errors='ignore')

    df.info()

    numeric_df = df.select_dtypes(include=['number'])
    print("\n📊 Check for Outliers")
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1

    outliers_iqr = df[((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(f"Number of outlier rows: {outliers_iqr.shape[0]}")
    print(outliers_iqr.head())  # print first few outliers

    df_clean = df[~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]
    numeric_df1 = df_clean.select_dtypes(include=['number'])
    plt.figure(figsize=(10,6))
    plt.boxplot(numeric_df)
    plt.xticks(range(1, len(numeric_df.columns)+1), numeric_df.columns, rotation=45)
    plt.title("Boxplot of Numeric Features")
    plt.show()
    print(f"Removed rows: {df.shape[0] - df_clean.shape[0]}")
    print(f"Remaining rows: {df_clean.shape[0]}")

    df_clean.reset_index(drop=True, inplace=True)
    return df_clean


# In[98]:


df_clean = clean_data(df)


# In[93]:


df_clean.info()


# In[ ]:





# In[ ]:




