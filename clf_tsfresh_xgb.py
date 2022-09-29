import numpy as np
import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

import datetime
import os
from utils import shap_feature_analysis, plot_cm, calcualte_classification_report
plt.ion()


def load_features_le(train_data_f, test_data_f, le_f):
    """Load precomputed features and label encoder

    Parameters
    ----------
    train_data_f: str
        Path to train data
    test_data_f: str
        Path to test data
    lef: str
        Path to label encoder

    Returns
    -------
    X_train: array-like
        Train data
    y_train: array-like
        Train labels
    X_test: array-like
        Test data
    y_test: array-like
        Test labels
    le: LabelEncoder
        Label encoder
    """

    # Load precomputed features
    train_data = np.loadtxt(train_data_f, delimiter=",")
    test_data = np.loadtxt(test_data_f, delimiter=",")

    # Preprocess data
    X_train = train_data[:,0:-1]
    y_train= train_data[:,-1].reshape(-1,1).ravel()
    X_test = test_data[:,0:-1]
    y_test = test_data[:,-1].reshape(-1,1).ravel()

    # Load label encoder
    with open(le_f, 'r') as f:
        mapping = json.load(f)
        le = LabelEncoder()

    nb_classes = len(mapping.keys())
    mapping['classes'] = [mapping[str(int(i))] for i in range(nb_classes)]
    le.classes_ = np.array(mapping['classes'])
    return X_train, y_train, X_test, y_test, le 

def main(train_data_f, test_data_f, output_dir, save_model, save): 
    """XGB Classifier, based on precomputed features"""

    le_f = "data/le_name_mapping.json"
    X_train, y_train, X_test, y_test, le = load_features_le(train_data_f, test_data_f, le_f)

    # Create XGBoost model
    model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Make predictions for train data
    y_train_pred = model.predict(X_train)
    plot_cm(y_train, y_train_pred, le, save=save, figname=f'{output_dir}/train_confusion')
    plt.close()
    # Make predictions for test data
    y_test_pred = model.predict(X_test)
    plot_cm(y_test, y_test_pred, le, save=save, figname=f'{output_dir}/test_confusion')
    plt.close()
    # Save XGB model to joblib
    if save_model:
        joblib.dump(model, f"{output_dir}/model.joblib")

    # Load feature names from json file to list
    with open('data/feature_names_tsfresh.json', 'r') as f:
        feature_names = json.load(f)
    
    # Create df from X_test and feature names
    df = pd.DataFrame(X_test, columns=feature_names)
    shap_feature_analysis(model, df, le, max_display=12, save=save, output_dir=output_dir)

    # Calculate f1 and save classification report
    calcualte_classification_report(y_train, y_train_pred, y_test, y_test_pred, le, save=save, output_dir=output_dir)
    print("Done")
    return

if __name__ == '__main__':
    remove_outlier = 0

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"results/clf/xgb/{now_str}"
    os.mkdir(output_dir)

    if remove_outlier:
        train_data_f = "data/train_tsfresh_ourem.csv"
        test_data_f = "data/test_tsfresh_ourem.csv"
        # Save outlier removed variable as txt file
        with open(f"{output_dir}/outlier_removed.txt", 'w') as f:
            f.write("Outlier removed")
    else:
        train_data_f = "data/train_tsfresh.csv"
        test_data_f = "data/test_tsfresh.csv"

    main(train_data_f, test_data_f, output_dir, save_model=True, save=True)