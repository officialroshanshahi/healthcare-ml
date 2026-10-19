import math
import numpy as np
from fastapi import status
import os, json, joblib, threading
from fastapi import BackgroundTasks, HTTPException
from sklearn.utils.validation import check_is_fitted
from src.v1.config import ACTIVE, VER
from src.v1.main import train_pipeline

# global state for locking different endpoints from acessing model
TRAINING_STATE = {"is_training": False}
state_lock = threading.Lock()

# function for checking if the model already exists or if it needs to be trained
def verify_and_get_model():
    
    global TRAINING_STATE
    
    # locking the model acquisition while in training
    with state_lock:
        if TRAINING_STATE["is_training"]:
            raise HTTPException(503, "Model is currently training in the backend. Please try again shortly.")
        
    # if the model is no longer in training
    if os.path.exists(ACTIVE):
        try:
            with open(ACTIVE) as f:
                a = json.load(f)
            
            ver, name = a["version"], a["model_name"]
            
            model = joblib.load(f"models/models/v{ver}/best_model_overall_{ver}.pkl")
            
            # checking if the model is trained or not
            check_is_fitted(model)
            
            print("model is trained and ready")
            
            return model, ver, name
        
        except Exception as e:
            print(f"[BACKEND] Model corrupted or unreadable: {e}")

    # state while model is training
    with state_lock:
        
        if TRAINING_STATE["is_training"]:
            raise HTTPException(503, "Model training already in progress.")
        
        TRAINING_STATE["is_training"] = True

    # this exception is then passed to handle_train_err which starts training pipeline
    raise HTTPException(status_code=503, detail="START_TRAINING")


# for starting training pipeline
def handle_train_err(e: HTTPException, bg_tasks: BackgroundTasks):
    
    # starting training pipeline if "START_TRAINING"
    if e.detail == "START_TRAINING":
        def _bg():
            try:
                train_pipeline(retrain='no', ver=VER)
            finally:
                with state_lock:
                    TRAINING_STATE["is_training"] = False
        
        bg_tasks.add_task(_bg)
        
        return {"status": "training", "message": "Pipeline initialized."}, status.HTTP_202_ACCEPTED
    
    raise e

# for extra safety changing nan to NONE
def clean_nan(obj):
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or np.isnan(obj)):
        return None
    return obj