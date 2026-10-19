import pandas as pd
from src.v1.preprocessing import y_preprocessor
from sklearn.model_selection import train_test_split

# Load the Parquet file directly into a pandas DataFrame
df_1 = pd.read_csv('data/raw/customer_churn_dataset-testing-master.csv')
df_2 = pd.read_csv('data/raw/customer_churn_dataset-training-master.csv')

df = pd.concat([df_1,df_2])

X = df.drop(columns='Churn').dropna()
y=df['Churn'].dropna()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2,stratify=y,random_state=42)

json_file = 'data/processed/v1/x_test_raw.json'

_test = pd.concat([x_test, y_test], axis=1)

_test.to_json(json_file, orient='records', indent=4)