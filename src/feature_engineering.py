"""
src/feature_engineering.py

This module contains functions for creating domain-informed structured features
from the raw dataset for the Fake Job Posting Detection System.
"""
import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Engineers 7 domain-informed structured features from the dataset.
    Supports both raw original columns and single 'text' column fallback.
    """
    df = df.copy()
    
    # Check if we are running in training/fallback mode (only 'text' column exists)
    is_fallback = 'text' in df.columns and ('description' not in df.columns or 'company_profile' not in df.columns)
    
    if is_fallback:
        # Scaffold all required columns for the preprocessor
        df['title'] = ''
        df['company_profile'] = ''
        df['description'] = df['text'].fillna('')
        df['requirements'] = ''
        df['benefits'] = ''
        
        df['employment_type'] = 'unspecified'
        df['required_experience'] = 'unspecified'
        df['required_education'] = 'unspecified'
        df['industry'] = 'unspecified'
        df['function'] = 'unspecified'
        
        df['telecommuting'] = 0
        df['has_company_logo'] = 0
        df['has_questions'] = 0
        df['salary_range'] = np.nan
        
        # Compute engineered features using fallback 'text' column
        df['description_length'] = df['text'].fillna('').str.len()
        df['company_profile_missing'] = 1  # Assume missing as we don't have separate field
        df['salary_missing'] = 1           # Assume missing as we don't have separate field
        
        pattern = r'\S+@\S+\.\S+|https?://'
        df['has_external_contact'] = df['text'].fillna('').str.contains(pattern, case=False, regex=True).astype(int)
        
        suspicious_keywords = [
            'urgent', 'urgently', 'deposit', 'commission', 'guaranteed', 
            'unlimited earnings', 'no experience required', 'work from home', 
            'training fee', 'registration fee', 'pay upfront', 'wire transfer'
        ]
        combined_text = df['text'].fillna('').astype(str).str.lower()
        keyword_pattern = '|'.join([r'\b' + kw.replace(' ', r'\s+') + r'\b' for kw in suspicious_keywords])
        df['suspicious_keyword_count'] = combined_text.str.count(keyword_pattern)
        
    else:
        # Standard full-column mode (inference and complete dataset mode)
        df['description_length'] = df['description'].fillna('').str.len()
        df['company_profile_missing'] = (df['company_profile'].isna() | (df['company_profile'].str.strip() == '')).astype(int)
        df['salary_missing'] = df['salary_range'].isna().astype(int)
        
        pattern = r'\S+@\S+\.\S+|https?://'
        df['has_external_contact'] = df['description'].fillna('').str.contains(pattern, case=False, regex=True).astype(int)
        
        suspicious_keywords = [
            'urgent', 'urgently', 'deposit', 'commission', 'guaranteed', 
            'unlimited earnings', 'no experience required', 'work from home', 
            'training fee', 'registration fee', 'pay upfront', 'wire transfer'
        ]
        text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        combined_text = df[text_columns].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
        
        keyword_pattern = '|'.join([r'\b' + kw.replace(' ', r'\s+') + r'\b' for kw in suspicious_keywords])
        df['suspicious_keyword_count'] = combined_text.str.count(keyword_pattern)
        
        df['has_company_logo'] = df['has_company_logo'].fillna(0).astype(int)
        df['has_questions'] = df['has_questions'].fillna(0).astype(int)
        
    return df

