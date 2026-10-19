import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from dotenv import load_dotenv

# project root and environment setup
project_root = Path(__file__).resolve().parents[2]
env_path = project_root / '.env'

load_dotenv(dotenv_path=env_path)

def get_resolved_path(env_var_name: str, fallback_path: str) -> Path:
    val = os.getenv(env_var_name, fallback_path)
    p = Path(val)
    return p if p.is_absolute() else (project_root / p).resolve()


# environment variables & paths (from .env)
ver = int(os.getenv("MODEL_VERSION", "1"))
VER = ver

PROD = str(get_resolved_path("PROD", "data/production/production_labels.csv"))
ACTIVE = str(get_resolved_path("ACTIVE", f"models/models/v{VER}/active_model.json"))
BASE = str(get_resolved_path("BASE_DATA", "data/raw/healthcare_dataset.csv"))
DB_FILE = str(get_resolved_path("DB_FILE", "users.db"))

# api
JWT_SECRET = os.getenv("JWT_SECRET", "change_this_default_secret_in_production")
LIMIT = int(os.getenv("RATE_LIMIT", "60"))
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
API_URL = os.getenv("API_URL", f"http://{HOST}:{PORT}")


# model constants
TARGET = "Test Results"
MIN_LABELS = 100         # For retraining
MIN_AUC = 0.55           # Used for training or retraining. it is .55 because of dataset. it isn't capable of higher auc

JWT_ALGORITHM = "HS256"
JWT_EXPIRY_SECONDS = 3600  # 1 hour

# only using RandomForestClassifier as it performed the best while eda
MODELS = {
    'RandomForestClassifier':
            # could have used class_weight="balanced" if it were imbalanced
            # but from all that plots it become pretty clear that the dataset is pretty evenly distributed
            RandomForestClassifier(random_state=42)
}

# hyperparameter tuning grid for XGBClassifier
PARAMS = {
    'RandomForestClassifier': {
            'n_estimators':[100, 200, 300, 500],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 5, 8, 12, 16, 20],
            'min_samples_split':[2, 5, 10, 20], # checking if higher sample split (default 2) creates better model
            'min_samples_leaf':[1, 2, 4, 8],
            'max_features': ['sqrt', 'log2'],
            'class_weight': [None, 'balanced', 'balanced_subsample']
    }
}

# scoring metrics for model evaluation
SCORES = {
    'accuracy': 'accuracy',
    'f1': 'f1_macro',
    'roc_auc': 'roc_auc_ovr',
    'precision': 'precision_macro',
    'recall': 'recall_macro'
}

# model monitoring
PSI_LIMIT = 0.20
KS_LIMIT = 0.05
MISSING_LIMIT = 0.10
PRED_LIMIT = 0.10
RESULTS_LIMIT = 0.10


# for getting base data
def get_base_data(ver_val=VER):
    """Dynamic getter for baseline data to avoid heavy execution on import."""
    import src.v1.preprocessing as pre
    return pre.load_data(ver=ver_val, base_only='yes')