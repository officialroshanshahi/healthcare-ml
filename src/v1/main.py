import json
import pandas as pd
import numpy as np
import uuid
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.v1.preprocessing import load_data, preprocessor
from src.v1.train import trainer
from src.v1.evaluate import evaluator
from src.v1.config import (
    TARGET, MIN_AUC, BASE, ACTIVE, MODELS, 
    PARAMS, SCORES, PROD, ver, MIN_LABELS
)

# project/src/v1/main.py -> parents[2] is project/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_resolved_path(path_str: str) -> Path:
    """safely resolves relative config paths against the project root."""
    path_obj = Path(path_str)
    return path_obj if path_obj.is_absolute() else (PROJECT_ROOT / path_obj).resolve()


def train_pipeline(ver=1, retrain='no', data=None):
    if retrain == 'yes' and data is not None:
        X = data.drop(columns=[TARGET])
        y = data[TARGET]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    else:
        Xtr, Xte, ytr, yte = load_data(ver=ver, base_only='no')

    # preprocessing
    X_train_processed, X_train_std = preprocessor(Xtr, "yes", "yes", ver)
    X_test_processed, X_test_std = preprocessor(Xte, "no", "yes", ver)

    # training
    models = trainer(X_train_processed, X_train_std, ytr, MODELS, PARAMS, SCORES, ver)

    # evaluation (set plot='yes' so evaluator returns in-memory PIL figures alongside metric dicts)
    best_model_name, best_info, eval_plots_list = evaluator(X_test_processed, X_test_std, yte, models, plot='yes', ver=ver)
    
    test_metrics = best_info['test_metrics']

    result = {
        "version": int(ver),
        "model_name": best_model_name,
        "accuracy": None if pd.isna(test_metrics["accuracy"]) else float(test_metrics["accuracy"]),
        "f1": None if pd.isna(best_info["cv_f1"]) else float(best_info["cv_f1"]),
        "roc_auc": None if pd.isna(test_metrics["roc_auc"]) else float(test_metrics["roc_auc"]),
        "precision": None if pd.isna(test_metrics["precision"]) else float(test_metrics["precision"]),
        "recall": None if pd.isna(test_metrics["recall"]) else float(test_metrics["recall"])
    }

    # if the model's AUC score is below the minimum threshold, reject it
    if float(result["roc_auc"]) < MIN_AUC:
        print(f"Model rejected: AUC {result['roc_auc']:.3f} < {MIN_AUC}")
        return {"status": "model didn't cross the threshold", **result}

    print(f"Model trained: AUC {result['roc_auc']:.3f} >= {MIN_AUC}")

    active_path = get_resolved_path(ACTIVE)
    active_path.parent.mkdir(parents=True, exist_ok=True)

    with open(active_path, "w") as f:
        json.dump(result, f, indent=2)

    return {"status": "model is completely trained and functional", **result}


def retrain(ver=1):
    prod_p = get_resolved_path(PROD)
    base_p = get_resolved_path(BASE)
    active_p = get_resolved_path(ACTIVE)

    if not prod_p.exists():
        return {"status": "pending", "reason": f"Missing file: {prod_p}"}

    prod = pd.read_csv(prod_p).dropna(subset=[TARGET])
    
    if TARGET not in prod.columns:
        return {"status": "pending", "reason": "No verified labels"}
    
    if len(prod) < MIN_LABELS:
        return {"status": "pending", "reason": f"Need {MIN_LABELS} labels; got {len(prod)}"}

    cols = [c for c in pd.read_csv(base_p).columns if c != TARGET]
    valid = [c for c in cols + [TARGET] if c in prod.columns]

    old_ver = ver
    if active_p.exists():
        with open(active_p, "r") as f:
            old_ver = json.load(f).get("version", 1)

    result = train_pipeline(ver=old_ver + 1, retrain='yes', data=prod[valid])

    if result.get("roc_auc", 0) < MIN_AUC:
        return {"status": "rejected", "reason": f"AUC {result.get('roc_auc', 0):.3f} < {MIN_AUC}", **result}

    print(f"New model promoted: AUC {result['roc_auc']:.3f} >= {MIN_AUC}")
    return {"status": "promoted", **result}


# prediction pipeline
def run_prediction(data: dict, m_inst):
    
    model, ver, name = m_inst
    
    # check model requirements before running the preprocessor to prevent unnecessary scaling execution overhead
    is_scaled_model = any(k in name.lower() for k in ["logisticregression", "kneighbor", "knclassifier"])
    
    # only compute and load the scaler transformations if the model specifically requires standard data
    if is_scaled_model:
        _, eval_data = preprocessor(pd.DataFrame([data]), "no", "yes", ver)
    else:
        eval_data = preprocessor(pd.DataFrame([data]), "no", "no", ver)
    
    # grabbing the full matrix of probabilities across all 3 classes
    probabilities = model.predict_proba(eval_data)[0] 
    
    # predicting by selecting the class with the highest probability value (Argmax method)
    predicted_class = int(np.argmax(probabilities))
    highest_prob = float(probabilities[predicted_class])
    prob_percentage = round(highest_prob * 100, 2)
    
    # map the numerical index back to the string label found in healthcare_dataset.csv
    inverse_label_map = {0: "Normal", 1: "Inconclusive", 2: "Abnormal"}
    predicted_label = inverse_label_map[predicted_class]
    
    # saving the production data log
    dt = pd.Timestamp.now().strftime("%Y-%m-%d")
    filepath = PROJECT_ROOT / f"data/new/v{ver}/production_data_{dt}.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    base_row = {
        "request_id": str(uuid.uuid4()),
        "timestamp": pd.Timestamp.now(),
        "prediction": predicted_label,
        "confidence_score": highest_prob,
        "confidence_percentage": f"{prob_percentage}%",
        "all_probabilities": str(probabilities.tolist())  # Saves full distribution data
    }
    
    # putting raw request values into base_row
    for k, val in data.items():
        base_row[k.lower()] = val
    
    df_row = pd.DataFrame([base_row])
    
    if filepath.exists(): 
        df_row = df_row.reindex(columns=pd.read_csv(filepath, nrows=0).columns, fill_value="")
        
    df_row.to_csv(filepath, mode="a", header=not filepath.exists(), index=False, quoting=1)
    
    return {
        "request_id": base_row["request_id"],
        "prediction": predicted_label,
        "confidence_score": highest_prob,
        "confidence_percentage": f"{prob_percentage}%",
        "model_version": ver,
        "model": name
    }

if __name__ == '__main__':
    result = train_pipeline(ver=1, retrain='no', data=None)