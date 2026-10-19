import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from PIL import Image

# explain_model function is designed to compute SHAP values for a given model and test dataset,
# generating visualizations and saving feature importance metrics
# it handles both single-sample and multi-sample scenarios,
# ensuring that the input data aligns with the model's expected features

def explain_model(x_test, model, name, explainer_type, ver=1, target_class=0):
    
    print(f"Calculating SHAP for {name}...")
    
    # ensure x_test is a DataFrame
    if not isinstance(x_test, pd.DataFrame):
        raise TypeError("x_test must be a pandas DataFrame")

    num_samples = len(x_test)
    
    X_sample = x_test.copy()

    # align feature names directly with model expectation
    expected_features = None
    
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        try:
            expected_features = model.get_booster().feature_names
        except Exception:
            pass
    
    # if expected features are known, ensure the test sample has the same columns
    if expected_features:
        
        missing_cols = [c for c in expected_features if c not in X_sample.columns]
        
        for c in missing_cols:
            X_sample[c] = 0.0
        
        X_sample = X_sample[expected_features]

    # if expected features are not known, use the columns from the test sample
    features = X_sample.columns.tolist()

    # build explicit background dataset to avoid E[f(X)] == f(x)
    # we load test_baseline or construct perturbation noise around sample
    project_root = Path(__file__).resolve().parents[2]
    baseline_path = project_root / f"data/processed/v{ver}/test_baseline.csv"
    
    if baseline_path.exists():
        
        bg_df = pd.read_csv(baseline_path)
        
        # filter out meta columns that are not features for SHAP calculations
        meta_cols = {
            "test_results",
            "request_id",
            "timestamp",
            "prediction",
            "probability",
            "risk_percentage",
            "risk_level"}
        
        # identify feature columns by excluding meta columns
        bg_features = [c for c in bg_df.columns if c not in meta_cols]
        
        # align background data with expected features if they are known
        bg_data = bg_df[bg_features]
        
        # align background features
        if expected_features:
            # ensure all expected features are present in the background data, adding missing ones as zeros
            for c in expected_features:
                if c not in bg_data.columns:
                    bg_data[c] = 0.0
                
            bg_data = bg_data[expected_features]
            
        bg_summary = shap.kmeans(bg_data, min(20, len(bg_data)))
    
    # if the baseline CSV does not exist, create a synthetic background dataset by adding noise to the test sample    
    else:
        # synthetic background noise if baseline CSV not found
        noise = np.random.normal(0, 0.1, size=(50, X_sample.shape[1]))
        bg_summary = pd.DataFrame(X_sample.values + noise, columns=features)

    # calculate SHAP values using KernelExplainer over model.predict_proba
    # forces evaluation directly on prediction probabilities with a distinct background population
    print("[SHAP INFO] Evaluating probabilities via KernelExplainer with population background...")
    
    # define a prediction function that returns the probability of the positive class
    # modified to return full multiclass probability matrix for all 3 classes
    predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=features))
    
    # initialize the SHAP KernelExplainer with the prediction function and background summary
    explainer = shap.KernelExplainer(predict_fn, bg_summary)
    
    # compute SHAP values for the test sample, handling both single and multiple samples
    vals = explainer.shap_values(X_sample)
    base = explainer.expected_value

    # extract target class specific arrays safely out of multiclass output structures
    if isinstance(vals, list):
        target_vals = vals[target_class]
    elif len(vals.shape) == 3:
        target_vals = vals[:, :, target_class]
    else:
        target_vals = vals

    if isinstance(base, (list, np.ndarray)):
        target_base = base[target_class]
    else:
        target_base = base

    # create a SHAP Explanation object to encapsulate the computed values, base values, and feature data
    shap_values = shap.Explanation(
        values=target_vals,
        base_values=target_base,
        data=X_sample.to_numpy(),
        feature_names=features
    )

    generated_images = []

    # helper function to convert matplotlib figures to PIL images for in-memory storage
    def fig_to_pil():
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return Image.open(buf)

    # beeswarm summary plot
    plt.figure()
    
    # if only one sample, use the background summary for beeswarm visualization to avoid misleading single-point representation
    if num_samples == 1:
        # evaluate background distribution samples to visualize point density
        bg_sample_eval = bg_summary.data if hasattr(bg_summary, "data") else bg_summary.iloc[:10]
        
        bg_vals = explainer.shap_values(bg_sample_eval)
        
        if isinstance(bg_vals, list):
            target_bg_vals = bg_vals[target_class]
        elif len(bg_vals.shape) == 3:
            target_bg_vals = bg_vals[:, :, target_class]
        else:
            target_bg_vals = bg_vals

        shap_batch = shap.Explanation(
            values=target_bg_vals,
            base_values=target_base,
            data=bg_sample_eval.to_numpy() if isinstance(bg_sample_eval, pd.DataFrame) else bg_sample_eval,
            feature_names=features
        )
        shap.plots.beeswarm(shap_batch, max_display=15, show=False)
    else:
        shap.plots.beeswarm(shap_values, max_display=15, show=False)

    
    plt.title(f"SHAP Summary (Beeswarm Class {target_class}) - {name}")
    plt.tight_layout()
    
    # save the beeswarm plot to the generated images list
    generated_images.append(fig_to_pil())

    # bar chart plot
    plt.figure()
    
    if num_samples == 1:
        shap.plots.bar(shap_values[0], max_display=15, show=False)
    else:
        shap.plots.bar(shap_values, max_display=15, show=False)
    
    plt.title(f"SHAP Feature Importance (Class {target_class}) - {name}")
    plt.tight_layout()
    
    generated_images.append(fig_to_pil())

    # waterfall plot
    plt.figure()
    
    # force single expression slicing for the waterfall plot calculation to bypass matrix shape constraints
    single_exp = shap_values[0] if num_samples == 1 else shap_values
    
    shap.plots.waterfall(single_exp, max_display=15, show=False)
    
    plt.title(f"SHAP Waterfall (Class {target_class}) - {name}")
    plt.tight_layout()
    
    generated_images.append(fig_to_pil())

    # save feature importance CSV
    out = Path(f"plots/shap/v{ver}")
    out.mkdir(parents=True, exist_ok=True)
    
    # calculate mean absolute SHAP values for each feature and save to CSV
    raw_vals = shap_values.values
    mean_abs = np.abs(raw_vals).mean(axis=0).ravel() if num_samples > 1 else np.abs(raw_vals).ravel()

    pd.DataFrame({
        "Feature": features,
        "Mean_Absolute_SHAP": mean_abs
    }).sort_values("Mean_Absolute_SHAP", ascending=False).to_csv(
        out / f"{name}_importance_{datetime.now():%Y%m%d_%H%M%S}.csv", index=False
    )

    
    print(f"{name} SHAP calculation completed successfully.")
    
    # return the list of generated PIL images for further use or display
    return generated_images