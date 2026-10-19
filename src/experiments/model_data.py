import os
from src.v1.preprocessing import preprocessor, y_preprocessor, load_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyarrow

ver = 1

#EXTRA CODE FOR EXPORTING DATASETS AND CHECKING CORRELATION
def export_processed_dataset(ver=1):

    X_train, X_test, y_train, y_test = load_data(ver=ver)

    #preprocessing x_train and x_test
    x_train_processed, x_train_processed_std = preprocessor(X_train, fit_transform='yes', std='yes', ver=ver)
    x_test_processed, x_test_processed_std = preprocessor(X_test, fit_transform='no', std='yes', ver=ver)

    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    
    #exporting them
    os.makedirs(f"data/processed/v{ver}", exist_ok=True)
    
    x_train_processed.to_parquet(f"data/processed/v{ver}/x_train_processed.parquet", index=False)
    x_train_processed_std.to_parquet(f"data/processed/v{ver}/x_train_processed_std.parquet", index=False)
    x_test_processed.to_parquet(f"data/processed/v{ver}/x_test_processed.parquet", index=False)
    x_test_processed_std.to_parquet(f"data/processed/v{ver}/x_test_processed_std.parquet", index=False)
    y_train.to_parquet(f"data/processed/v{ver}/y_train.parquet", index=False)
    y_test.to_parquet(f"data/processed/v{ver}/y_test.parquet", index=False)

    print(f'x_train_processed {x_train_processed.shape}')
    print('x_test_processed ',x_test_processed.shape)
    print('y_train ', y_train.shape)
    print('y_test ', y_test.shape)
    
    #for sanity check
    # print(x_train_processed.info())
    # print(x_train_processed_std.info())
    #print(x_test_processed.info())
    #print(x_test_processed_std.info())
    #print(x_train_processed.describe())
    #print(x_train_processed_std.describe())
    #print(x_test_processed.describe())
    #print(x_test_processed_std.describe())

    # ax = sns.heatmap(
    #     x_train_processed.corr(), 
    #     annot=True, 
    #     square=True, 
    #     fmt='.2f',
    #     linewidths=1, 
    #     linecolor='gray',
    #     annot_kws={"size": 8}
    # )

    # plt.xticks(
    #     rotation=45, 
    #     ha='right', 
    #     rotation_mode='anchor',
    #     fontsize=10
    # )

    # plt.yticks(fontsize=10)

    # plt.tight_layout()

    # os.makedirs(f"plots/features/v{ver}", exist_ok=True)
    
    # plt.savefig(f"plots/features/v{ver}/feature_correlation.png", bbox_inches="tight", dpi=300)
    # plt.show()

export_processed_dataset(ver=ver)