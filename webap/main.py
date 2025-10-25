from fastapi import FastAPI
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.dataload import load_data, data_summary, clean_data, scale
import pandas as pd
import numpy as np

app = FastAPI(title="DataReveal API", description="Backend for data loading and analysis", version="1.0")

@app.on_event("startup")
def startUp_event():
    global df
    df = load_data()
    print("✅ Data loaded successfully!")

@app.get("/")
def home():
    """message": "Welcome to DataReveal FastAPI"""
    """Data Summary"""
    return data_summary(df)

@app.get("/clean")
def clean():
    """Data summary before cleaning data"""
    summary_before = data_summary(df)
    global df_clean
    df_clean = clean_data(df)
    """Data summary after cleaning data"""
    summary_after = data_summary(df_clean)
    return {
        "message": "Welcome to DataReveal FastAPI",
        "summary_before_cleaning": summary_before,
        "summary_after_cleaning": summary_after,
        "df_clean": df_clean.head().to_dict(orient="records")
    }

@app.get("/scale")
def scalee():
    """Data Scaling after cleaning data"""
    try:
        result = scale(df_clean)
    except NameError:
        return {"error": "Please clean the data first by visiting /clean"}
    
    return result

@app.get("/preview")
def get_preview(limit: int = 5):
    """Get top rows of the dataset."""
    return df.head(limit).to_dict(orient="records")