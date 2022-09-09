from os import access
from re import A
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression



def baseline_model(train_data_f, test_data_file):
    """Baseline model for predicting ECM from EIS data
    Note: Models from the raw data performed very poorly, so we use the features extracted from the EIS data
    using tsfresh. For the baseline model we only use a few features and a linear classifier.

    """
    # Load data
    train_data = np.loadtxt(train_data_f, delimiter=",")
    test_data = np.loadtxt(test_data_file, delimiter=",")

    # Preprocess data
    X_train = train_data[:,0:-1]
    y_train= train_data[:,-1].reshape(-1,1)
    X_test = test_data[:,0:-1]
    y_test = test_data[:,-1].reshape(-1,1)

    # Make matrix Y of ECM labels 
    Y = np.zeros((y_train.shape[0], 9))
    for i in range(y_train.shape[0]):
        Y[i, int(y_train[i,0])] = 1
    
    # Train PLS DA model
    plsda = PLSRegression(n_components=2)
    plsda.fit(X_train, Y)
    y_pred = plsda.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)

    # Train RF model on the raw data

    # Print results

    return accs 


if __name__ == "__main__":
    train_data_f = "XGB/data/train_data_interpolated.csv"
    test_data_f = "XGB/data/test_data_interpolated.csv"
    accs = baseline_model(train_data_f, test_data_f)
    print(f'Accuracie raw data: {accs}')

    accs_umap = baseline_model()
    print(f'Accuracie UMAP data: {accs_umap}')