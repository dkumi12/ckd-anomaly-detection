# Reusable feature engineering module for CKD dataset
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def clean_raw_data(df):
    """
    Cleans raw dataframe formatting issues before pipeline processing.
    Addresses issues from known-issues.md.
    """
    df = df.copy()
    
    # 1. Drop IDs to prevent data leakage
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'PatientID' in df.columns:
        df = df.drop(columns=['PatientID'])
        
    # 2. Fix the string representation of missing values ('?' and '\t?')
    df = df.replace(r'^\s*\?\s*$', np.nan, regex=True)
    df = df.replace(r'^\t', '', regex=True)
    
    # Clean up target column if it's the training set
    if 'classification' in df.columns:
        df['classification'] = df['classification'].replace({'ckd\t': 'ckd', 'notckd': 'notckd', 'ckd': 'ckd'})
        df['classification'] = df['classification'].map({'notckd': 0, 'ckd': 1})
        
    return df

def build_preprocessor():
    """
    Builds a scikit-learn preprocessing pipeline to impute missing values 
    and scale/encode features.
    """
    # Based on the kidney disease dataset structure
    numeric_features = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    categorical_features = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

    # Numeric pipeline: Impute missing with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute missing with mode, then One-Hot Encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine both into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor