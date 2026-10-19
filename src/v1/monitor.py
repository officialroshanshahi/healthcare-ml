import os, numpy as np, pandas as pd
from datetime import datetime
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from evidently import Report
from evidently.presets import DataDriftPreset
from src.v1.config import TARGET, PSI_LIMIT, KS_LIMIT, MISSING_LIMIT, PRED_LIMIT, RESULTS_LIMIT, MIN_AUC


def psi(ref, cur, bins=10):
    # creating bins edges based on the reference distribution
    # np.unique ensures that duplicate values don't create overlapping or invalid bins
    e = np.unique(np.percentile(ref, np.linspace(0, 100, bins + 1)))
    
    # not enough data (to generate at least 2 bins) to calculate PSI, return 0
    if len(e) < 3: return 0
    
    a = np.clip(np.histogram(cur, e)[0] / len(cur), .0001, None) # current distribution
    b = np.clip(np.histogram(ref, e)[0] / len(ref), .0001, None) # reference distribution
    
    # calculating the PSI using the formula: sum((b - a) * log(b / a))
    return float(np.sum((b - a) * np.log(b / a)))


# this function monitors the model's performance and data drift by comparing a reference dataset with a current dataset.
def monitor_model(reference, current, predictions=None, pred_probas=None, labels=None, version=1):
    
    alerts = []
    
    # while calculating PSI and KS, we ignore the following columns as they are not relevant for drift detection
    ignore_cols = {'request_id', 'timestamp', TARGET, 'prediction', 'probability', 'risk_percentage', 'confidence_score', 'confidence_percentage', 'all_probabilities', 'test_results'}
    cols = [c for c in reference if c in current and c not in ignore_cols]
    
    ref, cur = reference[cols], current[cols]

    # sample a subset of the reference
    # and current datasets to limit the number of rows used for drift calculations, improving performance.
    sample_size = min(len(ref), len(cur), 20000)
    
    ref_sampled = ref.sample(n=sample_size, random_state=42)
    cur_sampled = cur.sample(n=sample_size, random_state=42)

    # check for missing values and data type mismatches in the sampled datasets
    for x in cols:
        
        miss = cur_sampled[x].isna().mean()
        
        if miss > MISSING_LIMIT: # currently set to 10% missing values threshold
            alerts.append(f"Missing: {x} {miss:.1%}")
        
        if not pd.api.types.is_numeric_dtype(ref_sampled[x]): # check if the reference column is numeric
            continue
        
        a, b = ref_sampled[x].dropna(), cur_sampled[x].dropna()
        
        # if either the reference or current column has fewer than 20 non-missing values, skip the drift calculations for that column
        if len(a) < 20 or len(b) < 20: 
            continue
        
        # calculate the p-value from the Kolmogorov-Smirnov test and the PSI for the current column
        p = ks_2samp(a, b).pvalue
        s = psi(ref_sampled[x], cur_sampled[x])
        
        # currently set to 20% PSI threshold for drift detection
        if s > PSI_LIMIT: 
            alerts.append(f"PSI drift: {x} {s:.3f}")
            
        elif p < KS_LIMIT:
            print(f"[MONITORING WARNING] Subtle KS variance on {x} (p={p:.4f}), bypassing automated retrain trigger.")

    # if predictions are provided,
    # check for prediction drift by comparing the mean of the predictions with the mean of the reference predictions
    if predictions is not None and "prediction" in reference:
        ref_counts = pd.Series(reference["prediction"]).value_counts(normalize=True)
        cur_counts = pd.Series(predictions).value_counts(normalize=True)
        all_classes = set(ref_counts.index).union(set(cur_counts.index))
        d = max([abs(ref_counts.get(c, 0) - cur_counts.get(c, 0)) for c in all_classes])
        if d > PRED_LIMIT: alerts.append(f"Prediction drift: {d:.2%}") # currently set to 10% prediction drift threshold for retraining trigger

    has_valid_labels = labels is not None and not pd.Series(labels).isna().all()
    
    auc = None
    
    # if valid labels are provided,
    # calculate the AUC score
    # and check if it falls below the minimum AUC threshold for retraining
    if has_valid_labels and pred_probas is not None:
        try:
            valid_mask = ~pd.Series(labels).isna()
            auc = roc_auc_score(np.array(labels)[valid_mask], np.array(pred_probas)[valid_mask], multi_class='ovr', average='macro')
            if auc < MIN_AUC: alerts.append(f"AUC: {auc:.3f}") # currently set to 0.6 AUC threshold for retraining trigger
        except ValueError: pass

    # if valid labels are provided,
    # check for churn drift by comparing the mean of the labels with the mean of the reference
    if has_valid_labels and ("test_results" in reference or TARGET in reference):
        t_col = "test_results" if "test_results" in reference else TARGET
        valid_labels = pd.Series(labels).dropna()
        ref_target_counts = pd.Series(reference[t_col]).value_counts(normalize=True)
        cur_target_counts = valid_labels.value_counts(normalize=True)
        all_targets = set(ref_target_counts.index).union(set(cur_target_counts.index))
        d = max([abs(ref_target_counts.get(c, 0) - cur_target_counts.get(c, 0)) for c in all_targets])
        if d > RESULTS_LIMIT: alerts.append(f"Results drift: {d:.2%}") # currently set to 10% churn drift threshold for retraining trigger

    path = ""
    
    # generate a monitoring report using the Evidently library,
    # which provides a visual summary of the data drift and model performance metrics.
    try:
        report = Report(metrics=[DataDriftPreset()])

        ev_ref = reference[cols].copy()
        ev_curr = current[cols].copy()

        # if valid labels are provided, add them to the reference and current datasets for the report
        if has_valid_labels:
            t_col = "test_results" if "test_results" in reference else TARGET
            ev_ref[t_col] = pd.to_numeric(reference[t_col], errors='coerce').fillna(0).astype(int)
            
            curr_labels = pd.to_numeric(labels, errors='coerce')
            
            valid_idx = curr_labels.dropna().index
            
            ev_curr = ev_curr.loc[valid_idx]
            
            ev_curr[t_col] = curr_labels.loc[valid_idx].astype(int)

        # run the report and handle any potential TypeError exceptions that may arise due to column mapping issues
        try:
            snapshot = report.run(reference_data=ev_ref, current_data=ev_curr)
        except TypeError:
            
            config_mapping = {
                "numerical_features": [c for c in cols if pd.api.types.is_numeric_dtype(reference[c])],
                "categorical_features": [c for c in cols if not pd.api.types.is_numeric_dtype(reference[c])]
            }
            
            if has_valid_labels:
                config_mapping["target"] = "test_results" if "test_results" in reference else TARGET
            
            snapshot = report.run(
                reference_data=ev_ref, 
                current_data=ev_curr, 
                column_mapping=config_mapping
            )
        
        # create a "reports" directory if it doesn't exist and save the monitoring report as an HTML file with a timestamped filename
        os.makedirs("reports", exist_ok=True)
        
        path = f"reports/monitor_v{version}_{datetime.now():%Y%m%d_%H%M%S}.html"
        
        snapshot.save_html(path)

    except Exception as e:
        print(f"Monitoring report generation warning: {e}")

    return {"report": path, "alerts": alerts, "retrain_required": bool(alerts), "auc": auc}