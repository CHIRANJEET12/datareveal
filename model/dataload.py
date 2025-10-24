#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[22]:


df = pd.read_csv("../csv/Mall_Customers.csv", na_values=["null"])


# In[23]:


df.head(), df.shape


# In[24]:


# df1 = pd.read_csv("csv/wine_data.csv")


# In[25]:


# df1.head()


# In[26]:


df


# In[27]:


def remid(df):
    m = df.columns[0]
    l = list(df.columns[0])
    l.reverse()
    s = ""
    for i in l:
        s+=i
        if s[:2].upper()=='DI':
            if m in df.columns:          
                df.drop(columns=[m], inplace=True)
                print(f"Dropped column: {m}")
            else:
                print(f"Column {m} not found, cannot drop.")
        else:
            print(f"Column {m} does not end with ID, not dropped.")

    return df


# In[28]:


remid(df)


# In[29]:


df.isnull().sum()


# In[30]:


df.duplicated().sum()


# In[31]:


numeric_df = df.select_dtypes(include=['int64','float64'])
numeric_df.corr()


# In[32]:


df.describe()


# In[33]:


df.dtypes


# In[34]:


sns.heatmap(numeric_df.corr())
plt.show()


# In[35]:


def dataunderstand(df):
    print("📊 Indepth Understanding of the dataset\n")

    print("1️⃣ First 5 rows:")
    print(df.head(), "\n")

    print("2️⃣ Dataset shape (rows, columns):")
    print(df.shape, "\n")

    print("3️⃣ Missing values in each column:")
    print(df.isnull().sum(), "\n")

    print("4️⃣ Number of duplicate rows:")
    print(df.duplicated().sum(), "\n")

    print("5️⃣ Statistical summary of numeric columns:")
    print(df.describe(), "\n")

    print("5️⃣ Correlation:")
    numeric_df = df.select_dtypes(include=['int64','float64'])
    numeric_df.corr()

    print("5️⃣ Heatmap:")
    sns.heatmap(numeric_df.corr())
    plt.show()

    print("5️⃣ Check for Outliers:")
    plt.figure(figsize=(10,6))
    plt.boxplot(numeric_df)
    plt.show()

    print("5️⃣ Histplot:")
    for c in numeric_df.columns:
        plt.figure(figsize=(13,8))
        sns.histplot(numeric_df[c], kde=True)



# In[36]:


dataunderstand(df)


# In[37]:


df.dropna()


# In[38]:


df.replace("null", np.nan, inplace=True)


# In[40]:


df.isnull()


# In[ ]:





# In[ ]:




