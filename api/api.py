from contextlib import asynccontextmanager
import os, glob, time, joblib, pandas as pd, uvicorn, io
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from anyio import to_thread
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from api.database.schema import UserAuth, ReqSchema, Batch, Label
from api.database.database import register_user, verify_user, init_db
from api.middleware.auth import auth, create_access_token
from api.middleware.train import verify_and_get_model, TRAINING_STATE, state_lock, handle_train_err
from api.controller.retrain import run_monitoring
from src.v1.preprocessing import preprocessor
from src.v1.main import run_prediction
from src.v1.config import ACTIVE, HOST, PORT
from src.v1.explain import explain_model
from src.v1.evaluate import eval_plots

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Healthcare Test Results API", lifespan=lifespan)

# dynamically checking current path
current_path = Path(__file__).resolve()
PROJECT_ROOT = next(p for p in current_path.parents if p.name == "healthcare-ml")

# for logging
L_DIR = PROJECT_ROOT / "logs"
L_FILE = L_DIR / "logs.csv"

# adding to the server log
def append_log_sync(log_row: dict):
    L_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([log_row]).to_csv(L_FILE, mode="a", header=not L_FILE.exists(), index=False)


# for logging everything i.e. no matter which endpoint user visits, this logs it
@app.middleware("http")
async def log_server_activity(request: Request, call_next):
    st = time.time()
    try: 
        res = await call_next(request)
        sc = res.status_code
    except Exception as e: 
        sc = 500; raise e
    finally:
        log = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "client_ip": request.client.host if request.client else "unknown",
            "method": request.method,
            "endpoint": request.url.path,
            "status_code": sc,
            "latency_seconds": round(time.time() - st, 4)}
        
        await to_thread.run_sync(append_log_sync, log)
        
    return res

"""
it is important to note that those end point logic could have been made into their own .py files
but for simplicity this is used
"""

@app.post("/register", status_code=201)
def register(user: UserAuth):
    if not register_user(user.username, user.password): 
        raise HTTPException(400, "Username registered.")
    return {"status": "success"}


@app.post("/login")
def login(u: UserAuth):
    if not verify_user(u.username, u.password): 
        raise HTTPException(401, "Invalid credentials.")
    return {"access_token": create_access_token(u.username), "token_type": "bearer"}


@app.get("/health")
async def health(b: BackgroundTasks, u=Depends(auth)):
    try: 
        _, version, model_name = verify_and_get_model()
        return {
            "status": "healthy", 
            "version": version, 
            "model": model_name
        }
    except HTTPException as e:
        return handle_train_err(e, b)


@app.post("/predict")
async def predict_api(r: ReqSchema, b: BackgroundTasks, _=Depends(auth)):
    try:
        return await to_thread.run_sync(run_prediction, r.data.dict(by_alias=True), verify_and_get_model())
    
    except HTTPException as e:
        return handle_train_err(e, b)


@app.post("/predict/batch")
async def batch(r: Batch, b: BackgroundTasks, _=Depends(auth)):
    try:
        return [await to_thread.run_sync(run_prediction, x.dict(by_alias=True), verify_and_get_model()) for x in r.data]
    except HTTPException as e:
        return handle_train_err(e, b)


# for labeling the production data based on request_id (you get on /predict or /predict/batch/)
@app.post("/label")
async def label(r: Label, _=Depends(auth)):
    
    def sync_l():
        _, ver, _ = verify_and_get_model()
        
        target_dir = PROJECT_ROOT / f"data/new/v{ver}"
        
        for f in glob.glob(os.path.join(target_dir, "production_data_*.csv")):
            
            d = pd.read_csv(f)
            
            # getting the request_id out
            mask = d.request_id.astype(str) == r.request_id
            
            # labeling the data
            if mask.any():
                d.loc[mask, "test results"] = r.test_results; d.to_csv(f, index=False); return {"status": "labeled"}
                
        raise HTTPException(404, "ID not found.")
    
    return await to_thread.run_sync(sync_l)


@app.post("/monitor")
async def monitor(_=Depends(auth)):
    try:
        model_instances = verify_and_get_model()
        
        results = await to_thread.run_sync(run_monitoring, model_instances)
        
        return {
            "status": "success",
            "message": "Monitoring analysis completed successfully.",
            "data_summary": results
        }
        
    except Exception as e:
        print(f"[API ROUTE ERROR] /monitor execution block failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Monitoring error: {str(e)}")


@app.get("/status")
def get_status():
    with state_lock:
        cur = TRAINING_STATE["is_training"]
    return {"status": "training" if cur else ("ready" if os.path.exists(ACTIVE) else "untrained")}


# def_sync could have added to its standalone file
# but for simplicity it is in api.py
@app.post("/explain", responses={200: {"content": {"image/png": {}}}})
async def explain(r: ReqSchema, b: BackgroundTasks, _=Depends(auth)):
    try: 
        model, version, name = verify_and_get_model()
    except HTTPException as e: 
        return handle_train_err(e, b)

    def sync_explain():
        
        # preprocessed data
        x, xs = preprocessor(pd.DataFrame([r.data.dict(by_alias=True)]), "no", "yes", version)
        
        # standard or not
        eval_data = xs if any(k in name.lower() for k in ["logistic", "kneighbor", "knc"]) else x
        
        # checking if the given eval data has any missing columns
        # if they do setting them to 0
        if hasattr(model, "feature_names_in_"):
            missing_cols = [c for c in model.feature_names_in_ if c not in eval_data.columns]
            
            for c in missing_cols:
                eval_data[c] = 0.0
            
            eval_data = eval_data[model.feature_names_in_]

        #converting the data as float for ease of use
        eval_data = eval_data.astype(float)

        # explainer based on the model type
        explainer = "tree" if any(k in name.lower() for k in ["forest", "tree", "boost"]) else ("linear" if any(k in name.lower() for k in ["linear", "logistic"]) else "kernel")
        
        # getting SHAP feature importance, waterfall and beeswarm (explicit target class index 0 passed)
        images = explain_model(x_test=eval_data, model=model, name=name, explainer_type=explainer, ver=version, target_class=0)
        
        # if no plots throwing an error
        if not images:
            raise HTTPException(500, "SHAP plot generation failed.")
        
        # from here is for sending those plots to user
        fig, axes = plt.subplots(len(images), 1, figsize=(12, 6 * len(images)))
        if len(images) == 1: 
            axes = [axes]
            
        for ax, img in zip(axes, images):
            ax.imshow(img)
            ax.axis('off')
            
        plt.tight_layout()
        
        img_buf = io.BytesIO()
        
        plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        
        plt.close(fig)
        
        img_buf.seek(0)
        
        return img_buf

    # running sync_explain to get the plots
    buffer = await to_thread.run_sync(sync_explain)
    
    # sending those plots so that it can be plotted in the frontend or swagger ui
    return StreamingResponse(buffer, media_type="image/png")


# same as /explain sync_eval() could have been made its own standalone file
# AND THIS IS EXTRA CODE THIS IS (IN MY OPINION) UNNECESSARY AS IT IS ALREADY DONE WHILE RUNNING TRAINING PIPELINE
@app.get("/evaluate", responses={200: {"content": {"image/png": {}}}})
async def evaluate_models(b: BackgroundTasks, _=Depends(auth)):
    try:
        model_inst = verify_and_get_model()
        ver = model_inst[1]
    except HTTPException as e:
        return handle_train_err(e, b)

    # for evaluating different models
    def sync_eval():
        
        metrics_path = PROJECT_ROOT / f"models/ev_metrics/v{ver}/best_models_metrics_{ver}.joblib"
        
        if not metrics_path.exists():
            raise HTTPException(404, detail="Evaluation metrics file not found. Please train models first.")
            
        best_models = joblib.load(metrics_path)
        
        images = eval_plots(best_models, ver=ver)

        if not images:
            raise HTTPException(500, detail="Evaluation plot generation failed.")

        fig, axes = plt.subplots(len(images), 1, figsize=(10, 5 * len(images)))
        if len(images) == 1:
            axes = [axes]
        for ax, img in zip(axes, images):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        img_buf.seek(0)
        return img_buf

    buffer = await to_thread.run_sync(sync_eval)
    
    return StreamingResponse(buffer, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=PORT, reload=True)