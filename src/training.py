"""
src/training.py

This module contains functions for instantiating, training, and serializing
the machine learning models for the Fake Job Posting Detection System.
"""
import time
import logging
from pathlib import Path
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline

from src.preprocessing import build_text_pipeline

logging.basicConfig(level=logging.INFO)

RANDOM_STATE = 42

def get_all_models():
    """Returns a dictionary of all 6 models specified in the proposal."""
    return {
        'dummy_stratified': DummyClassifier(strategy='stratified', random_state=RANDOM_STATE),
        'multinomial_nb': MultinomialNB(),
        'logistic_regression': LogisticRegression(
            C=1.0, penalty='l2', class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
        ),
        'linear_svc': CalibratedClassifierCV(
            LinearSVC(class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE),
            cv=5
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE
        ),
        'gradient_boosting': XGBClassifier(
            n_estimators=200, scale_pos_weight=20, use_label_encoder=False, 
            eval_metric='logloss', random_state=RANDOM_STATE
        )
    }

def build_mnb_pipeline():
    """Builds a separate pipeline for MultinomialNB that excludes the numeric/categorical branches."""
    return Pipeline([
        ('text_pipeline', build_text_pipeline()),
        ('classifier', MultinomialNB())
    ])

def train_all_models(X_train_transformed, y_train, models_dict):
    """Loops through all models, calls .fit(), returns a dict of fitted models."""
    fitted_models = {}
    
    for name, model in models_dict.items():
        logging.info(f"Training {name}...")
        start_time = time.time()
        
        try:
            model.fit(X_train_transformed, y_train)
            fitted_models[name] = model
            elapsed = time.time() - start_time
            logging.info(f"Finished {name} in {elapsed:.2f} seconds.")
        except Exception as e:
            elapsed = time.time() - start_time
            logging.error(f"Failed to train {name} after {elapsed:.2f} seconds. Error: {e}")
            
    return fitted_models

def cross_validate_model(pipeline, X_train, y_train):
    """Runs 5-fold stratified cross-validation on the TRAINING set only."""
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scorer = make_scorer(f1_score, pos_label=1)
    scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring=scorer, n_jobs=-1)
    
    mean_f1 = np.mean(scores)
    std_f1 = np.std(scores)
    
    logging.info(f"CV Fraud-Class F1: {mean_f1:.4f} (+/- {std_f1:.4f})")
    return mean_f1, std_f1

def save_artifacts(pipeline, threshold, vectorizer=None):
    """Serializes the final pipeline and threshold using joblib."""
    import joblib
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(pipeline, models_dir / 'final_pipeline.joblib')
    joblib.dump({'best_threshold': threshold}, models_dir / 'threshold.joblib')
    if vectorizer is not None:
        joblib.dump(vectorizer, models_dir / 'vectorizer_legacy.joblib')
        
def load_artifacts():
    """Loads the serialized pipeline and threshold using joblib."""
    import joblib
    models_dir = Path(__file__).parent.parent / 'models'
    pipeline = joblib.load(models_dir / 'final_pipeline.joblib')
    meta = joblib.load(models_dir / 'threshold.joblib')
    return pipeline, meta['best_threshold']
