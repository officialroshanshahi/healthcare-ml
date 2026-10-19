import pandas as pd
import numpy as np
from src.v1.preprocessing import load_data
from sklearn.model_selection import train_test_split

_, x_test, _, y_test = load_data()

y_test_df = pd.DataFrame(y_test, columns=['Test Results'], index=x_test.index).astype(int)

print(y_test_df.isnull().sum())

files = 'for_testing_model.json'

_test = pd.concat([x_test, y_test_df], axis=1)

_test.to_json(files, orient='records', indent=4)