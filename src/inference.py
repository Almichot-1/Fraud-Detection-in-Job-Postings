"""
src/inference.py

This module contains functions for generating risk signals and performing
inferences on single job postings using the pre-trained ML pipeline.
"""
import time
import logging
import pandas as pd
from src.feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO)

def get_risk_signals(input_dict, prediction_proba, threshold):
    """Evaluates engineered features on the input and returns risk signals."""
    df = pd.DataFrame([input_dict])
    df_engineered = engineer_features(df)
    
    signals = []
    
    if df_engineered['company_profile_missing'].iloc[0] == 1:
        signals.append("⚠️ No company profile provided — common in fraudulent postings")
        
    if df_engineered['salary_missing'].iloc[0] == 1:
        signals.append("⚠️ Salary range not disclosed")
        
    if df_engineered['has_external_contact'].iloc[0] == 1:
        signals.append("⚠️ External contact details detected in job description")
        
    kw_count = df_engineered['suspicious_keyword_count'].iloc[0]
    if kw_count > 0:
        signals.append(f"⚠️ {kw_count} suspicious keywords found")
        
    desc_len = df_engineered['description_length'].iloc[0]
    if 0 < desc_len < 200:
        signals.append(f"⚠️ Very short job description ({desc_len} characters)")
        
    if df_engineered['has_company_logo'].iloc[0] == 0:
        signals.append("⚠️ No company logo provided")
        
    if df_engineered['has_questions'].iloc[0] == 0:
        signals.append("⚠️ No screening questions provided")
        
    return signals

def predict_single(input_dict, pipeline, threshold):
    """Takes a single-row dict, engineers features, applies the trained pipeline, returns results."""
    start_time = time.time()
    
    df = pd.DataFrame([input_dict])
    df_engineered = engineer_features(df)
    
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(df_engineered)[0, 1]
    else:
        proba = float(pipeline.predict(df_engineered)[0])
        
    label = 'HIGH RISK' if proba >= threshold else 'LOW RISK'
    
    signals = get_risk_signals(input_dict, proba, threshold)
    
    execution_time = time.time() - start_time
    assert execution_time < 2.0, f"Inference exceeded 2 seconds ({execution_time:.2f}s)"
    
    return {
        'label': label,
        'probability': float(proba),
        'signals': signals
    }
