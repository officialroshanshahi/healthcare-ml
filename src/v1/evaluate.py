import datetime
import io
import joblib
import pandas as pd
import matplotlib
import numpy as np

# set the backend to 'Agg' for non-interactive environments (like servers or CI/CD pipelines)
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    precision_score, precision_recall_curve, 
    recall_score, roc_curve
)
from sklearn.preprocessing import label_binarize
from pathlib import Path

# Project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# helper function to generate evaluation plots and save them to disk
# helper function to generate evaluation plots and save them to disk
def eval_plots(best_models, ver=1):
    
    print("\nGenerating evaluation plots...")

    rows = []
    
    # iterate over the best models and their evaluation metrics to prepare a DataFrame for plotting
    for name, r in best_models.items():
        rows.append({
            "model": name,
            "cv_f1": r["cv_f1"],
            **r["test_metrics"]
        })

    # create a DataFrame from the collected metrics for easier plotting and saving
    df = pd.DataFrame(rows)
    
    # create a directory for evaluation plots if it doesn't exist
    eval_dir = PROJECT_ROOT / f"plots/evaluate/v{ver}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a unified timestamp for all files saved during this evaluation run
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # save evaluation metrics to CSV and Excel files
    df.to_csv(eval_dir / f"model_metrics_v{ver}_{timestamp}.csv", index=False)
    df.to_excel(eval_dir / f"model_metrics_v{ver}_{timestamp}.xlsx", index=False)

    generated_images = []

    # updated helper function to save to a specific file on disk AND return a PIL image
    def save_and_convert_fig(filename):
        # save the file to your output directory
        disk_path = eval_dir / f"{filename}_v{ver}_{timestamp}.png"
        plt.savefig(disk_path, format='png', dpi=150, bbox_inches='tight')
        
        # re-read into memory for the PIL list return step
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return Image.open(buf)

    # metrics comparison plot
    plt.figure(figsize=(10, 5))
    df.set_index("model")[["accuracy", "f1", "roc_auc", "precision", "recall"]].plot(
        kind="bar", figsize=(10, 5)
    )
    plt.title("Model Performance Comparison (Multiclass)")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=.3)
    plt.tight_layout()
    
    # save to disk as 'performance_comparison_...' and append to return list
    generated_images.append(save_and_convert_fig("performance_comparison"))

    # ROC curve (One-vs-Rest for 3 classes)
    plt.figure(figsize=(9, 6))
    for name, r in best_models.items():
        for class_idx, (fpr, tpr) in r["curves"]["roc"].items():
            plt.plot(fpr, tpr, label=f"{name} - Class {class_idx}")
            
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=.3)
    plt.tight_layout()
    
    # save to disk as 'roc_curves_...' and append to return list
    generated_images.append(save_and_convert_fig("roc_curves"))

    # Precision-Recall curve (One-vs-Rest for 3 classes)
    plt.figure(figsize=(9, 6))
    for name, r in best_models.items():
        for class_idx, (precision, recall) in r["curves"]["pr"].items():
            plt.plot(recall, precision, label=f"{name} - Class {class_idx}")
            
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (One-vs-Rest)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=.3)
    plt.tight_layout()
    
    # save to disk as 'pr_curves_...' and append to return list
    generated_images.append(save_and_convert_fig("pr_curves"))

    print(f"Metrics and plots successfully saved to: {eval_dir}/")
    print(df.round(4))
    
    return generated_images


# evaluator function to assess the performance of trained models on test data
def evaluator(x_test_processed, x_test_processed_std, y_test, trained_models, plot='no', ver=1):
    
    best_models = {}
    cv_scores = {}
    
    # load cross-validation scores if they exist, to avoid recalculating them
    cv_scores_path = PROJECT_ROOT / f"models/ev_search/v{ver}/cv_scores_{ver}.joblib"
    if cv_scores_path.exists():
        cv_scores = joblib.load(cv_scores_path)

    print("\nEvaluating models on test data...")

    # unique classes array for target binarization (e.g., [0, 1, 2])
    unique_classes = np.unique(y_test)
    n_classes = len(unique_classes)
    
    # binarize y_test cleanly for OvR curve calculation steps
    y_test_binarized = label_binarize(y_test, classes=unique_classes)

    # iterate over each trained model to calculate predictions, probabilities, and evaluation metrics
    for model_name, model in trained_models.items():
        
        m_lower = model_name.lower()
        
        # determine which test data to use based on the model type (standardized or not)
        if "logisticregression" in m_lower or "kneighborclassifier" in m_lower:
            X_te = x_test_processed_std
        else:
            X_te = x_test_processed

        y_pred = model.predict(X_te)

        # For multiclass ROC AUC, we require probabilities for ALL classes, not just index 1
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_te)
        else:
            # For linear estimators like LinearSVC that use decision_function
            y_probs = model.decision_function(X_te)
        
        # Dictionaries to hold structural OvR curves per class
        roc_curves_dict = {}
        pr_curves_dict = {}
        
        # Compute individual binary curves for each of the 3 classes
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_probs[:, i])
            precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_probs[:, i])
            
            roc_curves_dict[unique_classes[i]] = (fpr, tpr)
            pr_curves_dict[unique_classes[i]] = (precision, recall)

        # store the evaluation metrics and curves for each model in the best_models dictionary
        best_models[model_name] = {
            'model': model,
            'cv_f1': cv_scores.get(model_name, 0.0),
            'test_metrics': {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='macro'),
                'roc_auc': roc_auc_score(y_test, y_probs, multi_class='ovr', average='macro'),
                'precision': precision_score(y_test, y_pred, average='macro'),
                'recall': recall_score(y_test, y_pred, average='macro')
            },
            'curves': {
                'roc': roc_curves_dict,
                'pr': pr_curves_dict
            }
        }
    
    # save the best models metrics to a joblib file for future reference and analysis
    best_models_path = PROJECT_ROOT / f'models/ev_metrics/v{ver}/best_models_metrics_{ver}.joblib'
    best_models_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_models, best_models_path)

    # find best model based on cv_f1 score
    best_model_name, _ = max(
        best_models.items(), 
        key=lambda item: item[1]['cv_f1']
    )

    # retrieve the best model object for further use or saving to disk
    best_model = best_models[best_model_name]['model']
    print(f"\nBest model based on cross-validation F1 score: {best_model_name} with F1 score: {best_models[best_model_name]['cv_f1']:.4f}")
    
    # save the best model to disk for future use and deployment
    best_model_path = PROJECT_ROOT / f"models/models/v{ver}/best_model_overall_{ver}.pkl"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, best_model_path)

    
    # if the user requested plots, generate them and store them in the generated_plots list
    if str(plot).lower() == "yes":
        generated_plots = []
        generated_plots = eval_plots(best_models, ver=ver)
    
    # return the name of the best model, its evaluation metrics, and the list of generated plots (if any)
    return best_model_name, best_models[best_model_name], generated_plots