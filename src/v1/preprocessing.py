import numpy as np, pandas as pd, joblib
from pandas import get_dummies
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

# figuring out the project root directory dynamically, regardless of where the script is run from
# but since this file is in src/v1/, we can safely go up two levels (parents[2]) to reach the project root
# this can written in config, but this is a more robust way to ensure that the project root is always correctly identified
PROJECT_ROOT = Path(__file__).resolve().parents[2] if '__file__' in locals() else Path.cwd()

categorical_cols = ["gender", "blood_type", "medical_condition", "insurance_provider", "admission_type", "medication", "age_group"]

# creating a custom transformer for feature engineering
class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self): pass
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        
        x_copy = X.copy()
        
        # creating new features based on existing ones
        # new features such as hospital_res hospital_dur and hospital_bil were created but they had 0.9 corr to their respective y
        # so they were dropped
        
        x_copy['date_of_admission'] = pd.to_datetime(x_copy['date_of_admission'])
        x_copy['discharge_date'] = pd.to_datetime(x_copy['discharge_date'])
        
        x_copy['stay_duration'] = (x_copy['discharge_date'] - x_copy['date_of_admission']).dt.days
        
        x_copy['admission_month'] = x_copy['date_of_admission'].dt.month
        
        x_copy['admission_dayofweek'] = x_copy['date_of_admission'].dt.dayofweek
        
        # dropping unnecessary features
        x_copy.drop(columns=['date_of_admission', 'discharge_date'], inplace=True, errors='ignore')
        
        return x_copy
    
    def get_feature_names_out(self, input_features=None):
        
        if input_features is None: return None
        
        # we want to ensure that the new features are appended to the original feature list
        features = [f for f in input_features if f != 'test_results']
        
        new_features = ["stay_duration",
                        "admission_month",
                        "admission_dayofweek"]
        
        # we return the combined list of original and new features, ensuring that the new features are added at the end
        return np.array(features + new_features)


# renaming col to make things easier for us
def col_rename(x):
    x_copy = x.copy()
    
    x_copy.columns = [col.lower().replace(' ', '_') for col in x_copy.columns]
    
    # continuous age could have been used too but for simplicity i decided to use age bins
    age_bins = [0, 18, 40, 65, 120]
    age_labels = ['Pediatric', 'Young_Adult', 'Middle_Aged', 'Senior']
    
    x_copy['age_group'] = pd.cut(x_copy['age'], bins=age_bins, labels=age_labels)
    
    x_copy.drop(columns=['name', 'doctor', 'hospital', 'room_number', 'age'], errors='ignore', inplace=True)
    
    for col in categorical_cols:
        x_copy[col] = x_copy[col].astype('category')
    
    return x_copy


# get_dummies encoding with a dynamic column preservation step to handle single-sample inference
def encode(x):
    
    x_encoded = pd.get_dummies(x, columns=categorical_cols, drop_first=True, dtype=int)
    
    # we dynamically check if a tracking attribute exists on the function itself
    if not hasattr(encode, "training_columns"):
        
        # if it doesn't exist, this means we are currently fitting on the training set
        # save the full list of generated dummy columns directly to the function attribute
        encode.training_columns = x_encoded.columns.tolist()
        return x_encoded
        
    # force the incoming dataframe to match the structural columns of the training set
    # missing dummy columns are automatically re-created and populated with 0
    x_aligned = x_encoded.reindex(columns=encode.training_columns, fill_value=0)
    
    return x_aligned



# the main logic behind the preprocessing pipeline,
# which includes imputation, encoding, feature engineering, and scaling
# to avoid data leakage, we fit the pipeline on the training data and then transform the test data using the fitted pipeline
# use fit_transform='yes' for training data and fit_transform='no' for test data
def preprocessor(data, fit_transform='yes', std='yes', ver=1):
    
    data_copy = col_rename(data)
    
    # we will save the pipeline and scaler to disk for later use,
    # so we can apply the same transformations to new data during inference
    pipeline_path = PROJECT_ROOT / f'models/pipelines/v{ver}/pipeline_normal_v{ver}.joblib'
    scaler_path = PROJECT_ROOT / f'models/scalers/v{ver}/scaler_v{ver}.joblib'

    # if fit_transform is 'yes', we fit the pipeline and scaler on the training data and save them to disk
    if fit_transform.lower().strip() == 'yes':
        
        # imputer for the dataset
        impute_full = ColumnTransformer(transformers=[
            ('cat_', SimpleImputer(strategy='most_frequent'), make_column_selector(dtype_exclude='number', dtype_include='category')),
            ('num_', SimpleImputer(strategy='median'), make_column_selector(dtype_exclude='category', dtype_include='number'))
        ], verbose_feature_names_out=False, remainder='passthrough')

        
        pipeline_normal = Pipeline(steps=[
            ('imputer', impute_full),
            ('encoder', FunctionTransformer(encode)),
            ('feature_eng', CustomFeatureEngineer())])
        
        # we set the output of the pipeline to be a pandas DataFrame,
        # so we can easily access the feature names and work with the data
        pipeline_normal.set_output(transform="pandas")
        
        # we fit the pipeline on the training data and transform it, which will apply all the steps defined in the pipeline
        data_transformed = pipeline_normal.fit_transform(data_copy)
        
        # we save the fitted pipeline to disk so we can use it later to transform new data during inference (or for testing)
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline_normal, pipeline_path)

        print('Data preprocessing complete.\n')
        
        # checking if we need to scale the data,
        # if yes we will fit the scaler on the training data and returning both the transformed and scaled data
        if std.lower().strip() == 'yes':
            
            continuous_features = ['billing_amount', 'stay_duration', 'admission_month', 'admission_dayofweek']
            
            binary_features = [col for col in data_transformed.columns if col not in continuous_features and col != 'test_results']
            
            # we will scale only the numerical features,
            features_to_scale = [col for col in data_transformed.columns if col not in binary_features]
            
            scaler = ColumnTransformer(transformers=[
                ('scaler', StandardScaler(), features_to_scale)],
                remainder='passthrough',
                verbose_feature_names_out=False
            )
            
            
            scaler.set_output(transform="pandas")
            
            data_transformed_std = scaler.fit_transform(data_transformed)[data_transformed.columns]
            
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)
            
            print('Returning the processed dataset with scaling.\n')
            
            return data_transformed, data_transformed_std
        
        print('Returning the processed dataset without scaling.\n')
        
        return data_transformed
    
    # if fit_transform is 'no', we load the saved pipeline and scaler from disk and use them to transform the test data
    else:
        # ensuring that the pipeline exist before trying to load it, to avoid runtime errors
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found at {pipeline_path}.")
        
        pipeline_normal = joblib.load(pipeline_path)
        pipeline_normal.set_output(transform="pandas")
        
        data_transformed = pipeline_normal.transform(data_copy)
        
        print('Data preprocessing complete.\n')
        
        if std.lower().strip() == 'yes':
            # ensuring that the scaler file exists before trying to load it, to avoid runtime errors
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found at {scaler_path}.")
            
            scaler = joblib.load(scaler_path)
            
            scaler.set_output(transform="pandas")
            
            data_transformed_std = scaler.transform(data_transformed)[data_transformed.columns]
            
            print('Returning the processed dataset with scaling.\n')
            
            return data_transformed, data_transformed_std
        
        print('Returning the processed dataset without scaling.\n')
        
        return data_transformed


# preprocessing pipeline for the target variable (y), which includes imputation and type conversion
def y_preprocessor(y, ver=1):
    
    y_copy = y.copy()
    
    y_copy = (y_copy.map({'Normal': 0, 'Inconclusive': 1, 'Abnormal': 2}).values).astype('int8')
    
    return y_copy


# loading the raw data, splitting it into training and test sets, and returning them
# if base_only is 'yes', we return only the training set with the target variable (y) included,
# otherwise we return both training and test sets separately
# it is for monitoring purposes, where we want to have a baseline dataset to compare against the current data
def load_data(ver=1, base_only='no'):
    data = pd.read_csv(PROJECT_ROOT / "data/raw/healthcare_dataset.csv")
    data = data.drop_duplicates()
    
    X = data.drop(columns='Test Results')
    y = data['Test Results'].copy()
    
    y = y_preprocessor(y, ver=ver)
    
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        X = X.loc[y.index]
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    print("Data loading complete. Moving on to the next step in the pipeline.\n")

    if base_only == 'yes':
        return pd.concat([X_train, pd.Series(y_train, name='test_results', index=X_train.index)], axis=1)
    else:
        return X_train, X_test, y_train, y_test
