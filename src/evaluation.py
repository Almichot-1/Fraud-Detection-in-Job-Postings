"""
src/evaluation.py

This module contains functions for evaluating the models, calculating metrics,
optimizing classification thresholds, and generating visual plots.
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, 
    average_precision_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay
)

logging.basicConfig(level=logging.INFO)

def evaluate_model(model, X, y_true, model_name, threshold=0.5):
    """Evaluates the model and computes the specified metrics."""
    results = {}
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        results['pr_auc'] = average_precision_score(y_true, y_proba)
        results['roc_auc'] = roc_auc_score(y_true, y_proba)
    else:
        y_pred = model.predict(X)
        results['pr_auc'] = None
        results['roc_auc'] = None

    results['fraud_precision'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    results['fraud_recall'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    results['fraud_f1'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    results['classification_report'] = classification_report(y_true, y_pred, zero_division=0)
    
    return results

def optimize_threshold(model, X_val, y_val):
    """Iterates over threshold candidates on the validation set to maximize fraud-class F1."""
    if not hasattr(model, "predict_proba"):
        logging.warning("Model does not support predict_proba. Cannot optimize threshold.")
        return 0.5
        
    y_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        score = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        f1_scores.append(score)
        
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, label='F1 Score vs Threshold', color='blue')
    plt.axvline(x=best_threshold, color='red', linestyle='--', 
                label=f'Best Threshold = {best_threshold:.2f} (F1 = {f1_scores[best_idx]:.3f})')
    plt.title('Threshold Optimization on Validation Set')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Fraud-Class F1 Score')
    plt.legend()
    plt.grid(True)
    
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(models_dir / 'threshold_curve.png')
    plt.close()
    
    return best_threshold

def compare_all_models(results_dict):
    """Builds a DataFrame comparing all models on evaluated metrics."""
    records = []
    for model_name, metrics in results_dict.items():
        records.append({
            'Model': model_name,
            'Precision': metrics.get('fraud_precision', 0),
            'Recall': metrics.get('fraud_recall', 0),
            'F1': metrics.get('fraud_f1', 0),
            'PR-AUC': metrics.get('pr_auc', None),
            'ROC-AUC': metrics.get('roc_auc', None)
        })
        
    df = pd.DataFrame(records)
    df.sort_values(by='F1', ascending=False, inplace=True)
    
    logging.info(f"\nModel Comparison sorted by F1 (descending):\n{df.to_string(index=False)}")
    
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(models_dir / 'model_comparison.csv', index=False)
    
    return df

def plot_confusion_matrices(results_dict):
    """Creates a subplot grid of all confusion matrices from the evaluated models."""
    n_models = len(results_dict)
    if n_models == 0:
        return
        
    cols = 3 if n_models > 4 else 2
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for i, (model_name, metrics) in enumerate(results_dict.items()):
        cm = metrics.get('confusion_matrix')
        if cm is not None:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
            disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
            axes[i].set_title(f"{model_name}")
            
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(models_dir / 'confusion_matrices.png')
    plt.close()
