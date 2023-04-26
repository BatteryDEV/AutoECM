import numpy as np
import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import datetime
import os
import sys
from utils import shap_feature_analysis, plot_cm, calcualte_classification_report

plt.ion()


def load_features_le(train_data_f: str, test_data_f: str, le_f: str):
    """Load precomputed features and label encoder.

    Parameters
    ----------
    train_data_f: str
        Path to train data
    test_data_f: str
        Path to test data
    lef: str
        Path to label encoder
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
    X_train = train_data[:, 0:-1]
    y_train = train_data[:, -1].reshape(-1, 1).ravel()
    X_test = test_data[:, 0:-1]
    y_test = test_data[:, -1].reshape(-1, 1).ravel()

    # Load label encoder
    with open(le_f, "r") as f:
        mapping = json.load(f)
        le = LabelEncoder()

    nb_classes = len(mapping.keys())
    mapping["classes"] = [mapping[str(int(i))] for i in range(nb_classes)]
    le.classes_ = np.array(mapping["classes"])
    return X_train, y_train, X_test, y_test, le


def hyperparameter_optimization(
    le_f, cross_val="grid", save_best_params=True, output_dir=""
):
    """Cross validation for hyperparameter optimization"""

    X_train, y_train, X_test, y_test, le = load_features_le(
        train_data_f, test_data_f, le_f
    )
    if cross_val is not None:
        if cross_val == "grid":
            # Grid search implementation.
            parameters = {
                "max_depth": [5, 10, 20],
                "learning_rate": [0.1, 0.25],
                "subsample": [0.5, 0.75, 1.0],
                "colsample_bytree": [0.5, 0.75, 1.0],
                "n_estimators": [100, 375, 600],
            }

            clf = xgb.XGBClassifier(random_state=42)
            clf_s = GridSearchCV(
                clf, parameters, scoring="f1_macro", cv=4, n_jobs=-1, verbose=3
            )

        if cross_val == "random":
            parameters = {
                "max_depth": stats.randint(3, 25),
                "learning_rate": loguniform(1e-2, 0.9),
                "subsample": stats.uniform(0.3, 0.7),
                "colsample_bytree": stats.uniform(0.3, 0.7),
                "colsample_bylevel": stats.uniform(0.3, 0.7),
                "n_estimators": stats.randint(5, 750),
            }
            clf = xgb.XGBClassifier(random_state=42)
            clf_s = RandomizedSearchCV(
                clf,
                param_distributions=parameters,
                n_iter=250,
                scoring="f1_macro",
                cv=5,
                n_jobs=-1,
                verbose=3,
            )

        clf_s.fit(X_train, y_train)
        # Print the best parameters
        print("Best parameters set found on training set:")
        print(clf_s.best_params_)
        if save_best_params:
            with open(f"{output_dir}/best_hyperparameters.txt", "w") as f:
                f.write(str(clf_s.best_params_))
        tuned_model = clf_s.best_estimator_
        joblib.dump(clf_s, os.path.join(output_dir, "clf_s.pkl"))

    else:
        # tuned model found by previosu run, for comparison
        parameters = {
            'colsample_bylevel': 0.6918401578850566,
            'colsample_bytree': 0.5127192475374961,
            'learning_rate': 0.16527764697961278,
            'max_depth': 4,
            'n_estimators': 255,
            'subsample': 0.862351854582081,
        }
        tuned_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, **parameters)

    # Compare the tuned model with the default model by using the crossvalidation scores
    # IN the publication: XGB 1.5.0, default parameters: https://xgboost.readthedocs.io/en/latest/parameter.html
    default_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)

    tuned_model_score = cross_val_score(
        tuned_model, X_train, y_train, cv=5, n_jobs=-1, scoring="f1_macro"
    )
    default_model_score = cross_val_score(
        default_model, X_train, y_train, cv=5, n_jobs=-1, scoring="f1_macro"
    )
    print("Tuned model score: ", tuned_model_score.mean())
    print("Default model score: ", default_model_score.mean())
    print("Difference: ", tuned_model_score.mean() - default_model_score.mean())
    # Standard deviations
    print("Tuned model std: ", tuned_model_score.std())
    print("Default model std: ", default_model_score.std())
    print("Comparison done!")
    return tuned_model


def main(
    train_data_f,
    test_data_f,
    output_dir,
    le_f,
    save_model,
    model=None,
    cross_val=False,
    save=False,
):
    """XGB Classifier, based on precomputed features"""

    X_train, y_train, X_test, y_test, le = load_features_le(
        train_data_f, test_data_f, le_f
    )

    if model is None:
        model = xgb.XGBClassifier(random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    # Make predictions for train data
    y_train_pred = model.predict(X_train)
    plot_cm(y_train, y_train_pred, le, save=save, figname=f"{output_dir}/train_confusion")
    plt.close()
    # Make predictions for test data
    y_test_pred = model.predict(X_test)
    plot_cm(y_test, y_test_pred, le, save=save, figname=f"{output_dir}/test_confusion")
    plt.close()
    # Save XGB model to joblib
    if save_model:
        joblib.dump(model, f"{output_dir}/model.joblib")

    # if train_data_f contains filter then load the filtered features
    if "filtered" in train_data_f:
        with open("data/tsfresh/feature_names_tsfresh_filtered.json", "r") as f:
            feature_names = json.load(f)
    else:
        # Load feature names from json file to list
        with open("data/tsfresh/feature_names_tsfresh.json", "r") as f:
            feature_names = json.load(f)

    # Create df from X_test and feature names
    df = pd.DataFrame(X_test, columns=feature_names)
    shap_feature_analysis(model, df, le, max_display=7, save=save, output_dir=output_dir)

    # Calculate f1 and save classification report
    calcualte_classification_report(
        y_train, y_train_pred, y_test, y_test_pred, le, save=save, output_dir=output_dir
    )
    print("Done")
    return


if __name__ == "__main__":
    if len(sys.argv) < 3:
        filter = 0
        cross_val = None
        # "extended"  # "simple"
    else:
        filter = int(sys.argv[1])
        print(f"Filter: {filter}")
        cross_val = str(sys.argv[2])
        print(f"Cross validation: {cross_val}")

    le_f = "data/le_name_mapping.json"
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    if filter:
        train_data_f = "data/tsfresh/train_tsfresh_filtered.csv"
        test_data_f = "data/tsfresh/test_tsfresh_filtered.csv"
        output_dir = f"results/clf_filtered/xgb/{now_str}"
    else:
        train_data_f = "data/tsfresh/train_tsfresh.csv"
        test_data_f = "data/tsfresh/test_tsfresh.csv"
        output_dir = f"results/clf/xgb/{now_str}"

    os.mkdir(output_dir)
    if cross_val is not None:
        tuned_model = hyperparameter_optimization(
            le_f, cross_val=cross_val, save_best_params=True, output_dir=output_dir
        )

    else:
        # Tuned model, with random search. The filtered and unfitlered datasets are very similar
        # thus we are using the hyperparmeter values from the filtered dataset also for the unfiltered dataset.
        # Key observations from hyperparameter optimization:
        #   The changes in mode accuracy are relatively small.
        #   The default xgb model is already very good.
        parameters = {
            'colsample_bylevel': 0.6918401578850566,
            'colsample_bytree': 0.5127192475374961,
            'learning_rate': 0.16527764697961278,
            'max_depth': 4,
            'n_estimators': 255,
            'subsample': 0.862351854582081,
        }
        tuned_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, **parameters)

        main(
            train_data_f,
            test_data_f,
            output_dir,
            le_f,
            model=tuned_model,
            save_model=False,
            save=True,
        )

    print("Done")
