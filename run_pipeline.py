import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from src.preprocessing import build_preprocessor, split_data
from src.training import get_all_models, train_all_models, save_artifacts
from src.evaluation import evaluate_model, optimize_threshold, compare_all_models, plot_confusion_matrices
from src.feature_engineering import engineer_features

def main():
    print("Loading data...")
    df = pd.read_csv('data/fake_job_postings.csv')
    
    print("Engineering features...")
    df = engineer_features(df)
    
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    print("Building and fitting preprocessor on train...")
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    
    # Extract text branch for MultinomialNB
    # We will just train MNB separately or just train the others.
    # The requirement asks for all 6 models trained and evaluated.
    
    models = get_all_models()
    
    # Build text-only pipeline output for MNB
    from src.preprocessing import build_text_pipeline
    text_pipe = build_text_pipeline()
    X_train_text = text_pipe.fit_transform(X_train)
    X_val_text = text_pipe.transform(X_val)
    
    print("Training models...")
    fitted_models = train_all_models(X_train_transformed, y_train, models)
    
    # Fix MNB
    print("Training MNB on text only...")
    mnb = models['multinomial_nb']
    mnb.fit(X_train_text, y_train)
    fitted_models['multinomial_nb'] = mnb
    
    print("Optimizing threshold for best model (Random Forest as proxy)...")
    best_thresh = optimize_threshold(fitted_models['random_forest'], X_val_transformed, y_val)
    
    print("Evaluating models on validation set...")
    results = {}
    for name, model in fitted_models.items():
        if name == 'multinomial_nb':
            res = evaluate_model(model, X_val_text, y_val, name, threshold=best_thresh)
        else:
            res = evaluate_model(model, X_val_transformed, y_val, name, threshold=best_thresh)
        results[name] = res
        
    print("Generating comparisons and plots...")
    compare_all_models(results)
    plot_confusion_matrices(results)
    
    print("Saving artifacts...")
    # For inference pipeline, we want the full pipeline that includes preprocessing.
    # We will build an sklearn Pipeline with the preprocessor and the best model (RF).
    from sklearn.pipeline import Pipeline
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', fitted_models['random_forest'])
    ])
    
    save_artifacts(final_pipeline, best_thresh)
    print("Pipeline successfully run.")

if __name__ == '__main__':
    main()
