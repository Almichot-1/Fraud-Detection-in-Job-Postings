"""
src/preprocessing.py

This module handles the data preprocessing pipeline for the Fake Job Posting Detection System.
It includes custom transformers, ColumnTransformer configuration, and data splitting functions.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

TEXT_COLUMNS = ['title', 'company_profile', 'description', 'requirements', 'benefits']
CATEGORICAL_COLUMNS = ['employment_type', 'required_experience', 'required_education',
                       'industry', 'function']
BINARY_COLUMNS = ['telecommuting', 'has_company_logo', 'has_questions']
NUMERIC_COLUMNS = ['description_length', 'suspicious_keyword_count', 'company_profile_missing', 'salary_missing', 'has_external_contact']
TARGET_COLUMN = 'fraudulent'
RANDOM_STATE = 42

class TextConcatenator(BaseEstimator, TransformerMixin):
    """
    Custom transformer to concatenate specified text columns.
    Fills NaN with empty strings.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_df = pd.DataFrame(X)
        return X_df.fillna('').astype(str).agg(' '.join, axis=1)

def build_text_pipeline():
    """
    Returns a Pipeline containing the TextConcatenator and a TfidfVectorizer.
    """
    return Pipeline([
        ('concatenator', TextConcatenator()),
        ('tfidf', TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=3,
            max_features=50000,
            stop_words=None
        ))
    ])

def build_preprocessor():
    """
    Returns a ColumnTransformer mapping the three branches to their pipelines.
    """
    text_branch = build_text_pipeline()
    
    cat_branch = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unspecified')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    num_branch = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('text', text_branch, TEXT_COLUMNS),
        ('cat', cat_branch, CATEGORICAL_COLUMNS),
        ('num', num_branch, BINARY_COLUMNS + NUMERIC_COLUMNS)
    ], remainder='drop')
    
    return preprocessor

def split_data(df):
    """
    Performs a stratified three-way split: 70% train, 15% validation, 15% test.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    
    val_ratio = 15 / 85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=RANDOM_STATE
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
