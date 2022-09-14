from locale import normalize
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from utils import plot_cm
from hpsklearn import HyperoptEstimator, random_forest_classifier, standard_scaler
import xgboost as xgb

from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope
plt.ion()


def baseline_model(train_data_f, test_data_file, run_hyperopt=False):
    """Baseline model for predicting ECM from EIS data
    Linear classifiers perform poorly on this dataset, so we use a RF classifier (non-linear)
    Hyperparameter optimization with hpsklearn, keeping it simple for the baseline model
    Parameters
    ----------
    train_data_f : str
        Path to training data file
    test_data_file : str
        Path to test data file
    Returns
    -------
    None
    """
    # Load data
    train_data = np.loadtxt(train_data_f, delimiter=",")
    test_data = np.loadtxt(test_data_file, delimiter=",")

    # Preprocess data
    X_train = train_data[:,0:-1]
    y_train = train_data[:,-1].ravel()
    X_test = test_data[:,0:-1]
    y_test = test_data[:,-1].ravel()
    # Standardize data using sklearn standard scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def f1_loss_hp(y, y_pred):
        return 1 - f1_score(y, y_pred, average='macro')

    if run_hyperopt:
        # Restircitng the search space by setting some hyperparameters to default values
        clf = HyperoptEstimator(
            classifier=random_forest_classifier("Hyperopt RF", 
                            criterion='gini', class_weight='balanced_subsample', 
                            max_features='sqrt', bootstrap=True, oob_score=False, 
                            n_jobs=-1, random_state=42, verbose=0, warm_start=False, 
                            ccp_alpha=0.0, max_samples=None),
            preprocessing=[standard_scaler("Hyperopt Standard Scaler")], 
            n_jobs=-1, loss_fn=f1_loss_hp)
        clf.fit(X_train_scaled, y_train, n_folds=10, cv_shuffle=True, random_state=42)
        print(clf.score(X_test_scaled, y_test))
        print(clf.best_model())
        print("Done")
    else: 
        # Random Forest Classifier with hyperparamters from previous hyperparameter optimization run
        # New runs might give slightly different results
        clf = RandomForestClassifier(
            class_weight='balanced_subsample', n_estimators=150, n_jobs=-1, random_state=42)
        
        clf.fit(X_train_scaled, y_train)
    
    y_pred_test = clf.predict(X_test_scaled)
    y_pred_train = clf.predict(X_train_scaled)
    
    with open('data/le_name_mapping.json', 'r') as f:
        mapping = json.load(f)
        le = LabelEncoder()
    mapping['classes'] = [mapping[str(int(i))] for i in range(9)]
    le.classes_ = np.array(mapping['classes'])
    plot_cm(y_test, y_pred_test, le, save=1, figname='figures/rf_baseline/cm_rfb_test.png')
    plot_cm(y_train, y_pred_train, le, save=1, figname='figures/rf_baseline/cm_rfb_train.png')
    plt.show()

    acc_train = f1_score(y_train, y_pred_train, average='macro')
    acc_test = f1_score(y_test, y_pred_test, average='macro')

    print(f"F1 score train: {acc_train:.3f}")
    print(f"F1 score test: {acc_test:.3f}")
    print('Done')
    return 


if __name__ == "__main__":
    train_data_f = "data/train_data_newspl_inter.csv"
    test_data_f = "data/test_data_newspl_inter.csv"

    acc_train, acc_test = baseline_model(train_data_f, test_data_f, run_hyperopt=False)

    #accs_umap = baseline_model()
    #print(f'Accuracie UMAP data: {accs_umap}')