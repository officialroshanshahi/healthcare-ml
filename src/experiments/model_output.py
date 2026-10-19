import joblib
import pandas as pd
from ..v1.train import trainer
from scipy.stats import loguniform, uniform
from sklearn.ensemble import RandomForestClassifier
from src.v1.evaluate import evaluator

#EXTRA CODE
ver = 1

def train_models(ver=1):
    #importing necessary processed data and exporting the trained model - EXTRA CODE
    x_train_processed = pd.read_parquet(f"data/processed/v{ver}/x_train_processed.parquet")
    x_train_processed_std = pd.read_parquet(f"data/processed/v{ver}/x_train_processed_std.parquet")
    y_train = pd.read_parquet(f"data/processed/v{ver}/y_train.parquet").values.ravel()

    model_dic = {
      'RandomForestClassifier':
          # could have used class_weight="balanced" if it were imbalanced
          # but from all that plots it become pretty clear that the dataset is pretty evenly distributed
          RandomForestClassifier(random_state=42)
    }

    params ={
      'RandomForestClassifier': {
                  'n_estimators': [200, 300, 500, 700],
                  'criterion': ['gini', 'entropy', 'log_loss'],
                  'max_depth': [None, 5, 8, 12, 16, 20],
                  'min_samples_split': [2, 5, 10, 20], # checking if higher sample split (default 2) creates better model
                  'min_samples_leaf': [1, 2, 4, 8],
                  'max_features': ['sqrt', 'log2'],
                  'class_weight': [None, 'balanced', 'balanced_subsample']
          }
    }
    
    scoring_metrics = {
      'accuracy': 'accuracy',
      'f1': 'f1_macro',
      'roc_auc': 'roc_auc_ovr',
      'precision': 'precision_macro',
      'recall': 'recall_macro'
    }

    best_models = trainer(x_train_processed,
                          x_train_processed_std,
                          y_train,
                          model_dic=model_dic,
                          params=params,
                          scoring_metrics=scoring_metrics,
                          ver=ver
    )

# train_models(ver=ver)

def evaluate(ver=1):
  x_test_processed = pd.read_parquet(f"data/processed/v{ver}/x_test_processed.parquet")
  x_test_processed_std = pd.read_parquet(f"data/processed/v{ver}/x_test_processed_std.parquet")
  y_test = pd.read_parquet(f"data/processed/v{ver}/y_test.parquet").values.ravel()
  
  RandomForestClassifier = joblib.load(f'models/models/v{ver}/RandomForestClassifier_best_{ver}.pkl')
  
  trained_models = {
    'RandomForestClassifier': RandomForestClassifier
  }
  
  evaluator(x_test_processed, x_test_processed_std, y_test, trained_models, plot='yes', ver=1)
  
# evaluate()