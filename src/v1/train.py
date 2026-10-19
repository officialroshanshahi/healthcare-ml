import joblib, numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, train_test_split

# explicitly grab index [2] to get a clean Path object instead of a _PathParents sequence
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# trainer function that can train multiple models with hyperparameter tuning
# and save the best models and search (RandomizedSearchCV in this case) results to disk
# of course, search is not necessary, but it is useful for analysis and debugging
def trainer(x_train, x_train_pr, y_train, model_dic, params, scoring_metrics, ver):
    
    print('\nStarting training pipeline.\n')
    
    best_models, cv_scores = {}, {}
    
    # flatten y_train to ensure it is a 1D array for stratification and model fitting
    y_train_flat = y_train.squeeze()
    
    # split the training data into a training and validation set for hyperparameter tuning
    train, val = train_test_split(np.arange(len(x_train)), test_size=0.20, stratify=y_train_flat, random_state=42)
    
    # PredefinedSplit allows us to use the same training and validation split for all models during hyperparameter tuning
    split_fold = np.zeros(len(x_train))
    split_fold[train] = -1
    
    pds = PredefinedSplit(test_fold=split_fold)
    
    # iterate over each model in the model dictionary,
    # perform hyperparameter tuning using RandomizedSearchCV,
    # and save the best models and search results to disk
    for m_name, init_model in model_dic.items():
        
        print(f"Searching space for {m_name}...")
        
        m_low = m_name.lower()
        
        # use the scaled training data for models that require it (like Logistic Regression or KNeighbors),
        X_tr = x_train_pr.copy() if any(k in m_low for k in ["logisticregression", "kneighbor", "knclassifier"]) else x_train.copy()
        
        # this can be removed if you have enough computing power
        n_workers = 1
        
        fit_params = {}
        
        # for XGBClassifier, we set specific parameters for early stopping and evaluation metrics
        if "xgbclassifier" in m_low:
            
            init_model.set_params(tree_method="hist", eval_metric="logloss", early_stopping_rounds=15)
            
            fit_params = {"eval_set": [(X_tr.iloc[val], y_train_flat.iloc[val])], "verbose": False}

        search = RandomizedSearchCV(
            estimator=init_model,
            param_distributions=params[m_name],
            n_iter=10,
            cv=pds,
            scoring=scoring_metrics,
            refit="f1",
            random_state=42,
            n_jobs=n_workers)
        
        # fit the model using the training data and the predefined split for validation
        search.fit(X_tr, y_train_flat, **fit_params)
        
        best_model = search.best_estimator_

        # save the best model and the search results to disk for future reference or deployment
        for path, obj in [(f"models/models/v{ver}/{m_name}_best_{ver}.pkl", best_model),
                        (f"models/ev_search/v{ver}/{m_name}_search_{ver}.pkl", search)]:
            
            (PROJECT_ROOT / path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(obj, PROJECT_ROOT / path)

        best_models[m_name], cv_scores[m_name] = best_model, search.best_score_
        
        print(f"Search complete for {m_name}.")

    print("\nAll the models have been trained.")
    
    # save the cross-validation scores to disk for future reference or analysis
    (PROJECT_ROOT / f"models/ev_search/v{ver}/cv_scores_{ver}.joblib").parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cv_scores, PROJECT_ROOT / f"models/ev_search/v{ver}/cv_scores_{ver}.joblib")
    
    # this function can be made to return the best models
    # and their cross-validation scores (without saving search or cv_scores in the disk) for further evaluation or analysis
    # but for debugging and analysis, saving the search and cv_scores to disk is useful
    # or save only the cv_scores to disk and return the best models for further evaluation or analysis
    return best_models