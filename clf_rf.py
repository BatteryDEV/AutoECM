import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from typing import Literal, Optional, Tuple

# from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
# from hyperopt.pyll import scope
from utils import plot_cm, calcualte_classification_report
import pickle
import datetime
import os
import sys

np.random.seed(42)

plt.ion()


def tune_nb_estimator(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    folds: int = 5,
    params_dict: dict = {},
) -> Tuple[int, int]:
    n_estimators = [10, 50, 100, 150, 200, 500, 800, 1000]
    f1 = []
    f1_std = []

    # remove nb_estimators if they are already in params_dict
    if "n_estimators" in params_dict.keys():
        params_dict.pop("n_estimators")

    for nb in n_estimators:
        print(f"CV with number of estimators: {nb}")
        clf = RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators=nb,
            n_jobs=-1,
            random_state=42,
            **params_dict,
        )

        # Cross-validation: Predict the test samples based on a predictor that was trained with the
        # remaining data. Repeat until prediction of each sample is obtained.
        # (Only one prediction per sample is allowed)
        # Only these two cv methods work. Reson: Each sample can only belong to EXACTLY one test set.
        # Other methods of cross validation might violate this constraint
        # For more information see:
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html
        scores = cross_val_score(
            clf, X_train_scaled, y_train, cv=folds, n_jobs=-1, scoring="f1_macro"
        )
        f1.append(scores.mean())
        f1_std.append(scores.std())
    # Pick the best number of estimators based on the one standard deviation rule
    best_nb = n_estimators[np.argmax(f1)]
    print("Best number of estimators: {}".format(best_nb))

    # If you desire a smaller model, uncomment the following lines and use best_nb_std
    f1_max_loc = np.argmax(f1)
    filtered_lst = [
        (i, element)
        for i, element in enumerate(f1)
        if element > f1[f1_max_loc] - (1 * f1_std[f1_max_loc])
    ]
    f1_std_max_loc, _ = min(filtered_lst)
    best_nb_std = n_estimators[f1_std_max_loc]
    print("Best number of estimators (1std): {}".format(best_nb_std))
    return best_nb, best_nb_std


# Hyperopt for hyperparameter optimization did not work well for this dataset.
def baseline_model(
    train_data_f: str,
    test_data_file: str,
    output_dir: str,
    compare_models: bool = True,
    cross_val: Optional[Literal["simple", "extended"]] = None,
    filtered: bool = False,
    save: bool = False,
    save_model: bool = False,
) -> None:
    """Baseline model for predicting ECM from EIS data
    Linear classifiers perform poorly on this dataset, so we use a RF classifier (non-linear)
    Hyperparameter optimization with hpsklearn, keeping it simple for the baseline model
    Parameters
    ----------
    train_data_f : str
        Path to training data file
    test_data_file : str
        Path to test data file
    output_dir : str
        Path to output directory
    compare_models : bool
        Compare different models cross validation accouracies
    cross_val : bool
        Perform cross validation
    save : bool
        Save figures

    Returns
    -------
    acc_test : float
        Test accuracy
    acc_train : float
        Train accuracy
    """
    # Load data
    train_data = np.loadtxt(train_data_f, delimiter=",")
    test_data = np.loadtxt(test_data_file, delimiter=",")

    # Preprocess data
    X_train = train_data[:, 0:-1]
    y_train = train_data[:, -1].ravel()
    X_test = test_data[:, 0:-1]
    y_test = test_data[:, -1].ravel()
    # Standardize data using sklearn standard scaler
    # This approach does not make sense, Zreal and Zimag should not be scaled independently
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # The following ensures, that Zreal and Zimag are scaled together
    for i in range(X_train.shape[0]):
        divider = np.max(X_train[i, :30])
        X_train[i, :] = X_train[i, :] / divider
    X_train_scaled = X_train.copy()
    for i in range(X_test.shape[0]):
        divider = np.max(X_test[i, :30])
        X_test[i, :] = X_test[i, :] / divider
    X_test_scaled = X_test.copy()

    folds = 5
    if cross_val is not None:
        if cross_val == "simple":
            # Use sklearn 5 fold CV to determine the number of estimators
            # Loop through all possibilities
            best_nb, best_nb_std = tune_nb_estimator(X_train_scaled, y_train, folds)

            simple_model = RandomForestClassifier(
                class_weight="balanced_subsample",
                n_estimators=best_nb,
                n_jobs=-1,
                random_state=42,
            )

        if cross_val == "extended":
            folds = 5
            parameters = {
                "bootstrap": [True, False],
                "max_depth": [10, 75, None],
                "max_features": ["sqrt", None],
                "min_samples_leaf": [1, 3, 5, 10],
                "min_samples_split": [2, 3, 5, 8],
                "n_estimators": [10, 100, 300, 600],
            }

            clf = RandomForestClassifier(
                class_weight="balanced_subsample", max_depth=None, random_state=42
            )

            clf_gs = GridSearchCV(
                clf, parameters, scoring="f1_macro", cv=folds, n_jobs=-1, verbose=5
            )

            clf_gs.fit(
                X_train_scaled,
                y_train,
            )
            # Print the best parameters
            print("Best parameters set found on training set:")
            print(clf_gs.best_params_)
            clf = clf_gs.best_estimator_
            with open(f"{output_dir}/best_hyperparameters.txt", "w") as f:
                f.write(str(clf_gs.best_params_))

    if compare_models:
        # Models from previous runs
        simple_model = RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators=800,
            n_jobs=-1,
            random_state=42,
        )

        params_dict = {
            "bootstrap": True,
            "max_depth": 75,
            "max_features": None,
            "min_samples_leaf": 1,
            "min_samples_split": 3,
            "n_estimators": 600,
        }
        gs_model = RandomForestClassifier(
            class_weight="balanced_subsample", n_jobs=-1, random_state=42, **params_dict
        )

        # Compare the performance of the model with only the number of estimators tuned and the larger grid search
        gs_model_score = cross_val_score(
            gs_model, X_train, y_train, cv=5, n_jobs=-1, scoring="f1_macro"
        )
        default_model_score = cross_val_score(
            simple_model, X_train, y_train, cv=5, n_jobs=-1, scoring="f1_macro"
        )
        print("Grid search model score: ", gs_model_score.mean())
        print("Simple model score: ", default_model_score.mean())
        print("Difference: ", gs_model_score.mean() - default_model_score.mean())
        # Standard deviations
        print("Grid search model std: ", gs_model_score.std())
        print("Simple model std: ", default_model_score.std())
        print("Comparison done!")

    # Results in the manuscript are based on the following parameters:
    # Best parameters set found on filtered training set with grid search:
    # (The results where the same fot the filtered and unfiltered data)
    params_dict = {
        "bootstrap": True,
        "max_depth": 75,
        "max_features": None,
        "min_samples_leaf": 1,
        "min_samples_split": 3,
        "n_estimators": 300,
    }

    clf = RandomForestClassifier(
        class_weight="balanced_subsample", n_jobs=-1, random_state=42, **params_dict
    )

    score = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1, scoring="f1_macro")
    print(score)
    clf.fit(X_train_scaled, y_train)

    y_test_pred = clf.predict(X_test_scaled)
    y_train_pred = clf.predict(X_train_scaled)

    with open("data/le_name_mapping.json", "r") as f:
        mapping = json.load(f)
        le = LabelEncoder()
    mapping["classes"] = [mapping[str(int(i))] for i in range(9)]
    le.classes_ = np.array(mapping["classes"])
    plot_cm(y_test, y_test_pred, le, save=save, figname=f"{output_dir}/cm_rfb_test")
    plot_cm(y_train, y_train_pred, le, save=save, figname=f"{output_dir}/cm_rfb_train")
    plt.show()

    # Save model
    if save_model:
        with open(f"{output_dir}/rf.pkl", "wb") as f:
            pickle.dump(clf, f)

    # Calculate f1 and save classification report
    calcualte_classification_report(
        y_train, y_train_pred, y_test, y_test_pred, le, save=save, output_dir=output_dir
    )

    return


if __name__ == "__main__":
    if len(sys.argv) < 3:
        filter = 1
        cross_val = None
        # "extended"  # "simple"
    else:
        filter = int(sys.argv[1])
        print(f"Filter: {filter}")
        cross_val = str(sys.argv[2])
        print(f"Cross validation: {cross_val}")

    # Create new folder with results, name is datetime
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    if filter:
        train_data_f = "data/interpolated/train_data_filtered.csv"
        test_data_f = "data/interpolated/test_data_filtered.csv"
        output_dir = f"results/clf_filtered/rf/{now_str}"
    else:
        train_data_f = "data/interpolated/train_data.csv"
        test_data_f = "data/interpolated/test_data.csv"
        output_dir = f"results/clf/rf/{now_str}"

    os.mkdir(output_dir)

    baseline_model(
        train_data_f,
        test_data_f,
        output_dir,
        compare_models=False,
        cross_val=cross_val,
        filtered=filter,
        save=True,
        save_model=False,
    )
    print("Done")
