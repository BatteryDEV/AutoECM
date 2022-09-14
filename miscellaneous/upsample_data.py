import numpy as np
import pandas as pd
from sklearn.utils import resample

def resample_data(X, y, method='downsample', random_state=42):
    """Downsamples classification data using resampling with replacement.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,)
        Target values.

    method : str, optional (default='downsample')
        Method to use for resampling. Options are 'downsample' and 'upsample'.

    random_state : int, optional (default=42)
        Seed for random number generator.

    Returns
    -------
    X_resampled : array-like, shape (n_samples_new, n_features)
        Downsampled data.

    y_resampled : array-like, shape (n_samples_new,)
        Downsampled target values.
    """
    
    # Nb of samples
    nb_samples = [np.sum(y==i) for i in np.unique(y)]
    if method=='downsample':
        n_samples = np.min(nb_samples)
    elif method=='upsample':
        n_samples = np.max(nb_samples)
    else:
        raise ValueError("method must be either 'downsample' or 'upsample'")

    print(f"Nb of samples: {n_samples}")
    nb_classes = len(np.unique(y)) 

    X_resampled = np.zeros((n_samples*nb_classes, X.shape[1]))
    y_resampled = np.zeros(n_samples*nb_classes)

    # Downsample
    for i in np.unique(y):
        idx = np.where(y == i)[0]
        l_id = int(i*n_samples)
        u_id = int((i+1)*n_samples)
        X_resampled[l_id:u_id, :], y_resampled[l_id:u_id] = resample(
            X[idx, :], y[idx], n_samples=n_samples)

    return X_resampled, y_resampled


def main(): 
    # Read in data
    data_fname = "XGB/data/train_data_interpolated.csv"
    data = np.loadtxt(data_fname, delimiter=',')  

    test_data_fname = 'XGB/data/test_data_interpolated.csv'
    test_data = np.loadtxt(test_data_fname, delimiter=',')

    # Split into X and y
    X_train = data[:, :-1]
    y_train = data[:, -1]

    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    # Downsample
    print("Downsampling...")
    X_train_downsampled, y_train_downsamples = resample_data(X_train, y_train, method='downsample')
    X_test_downsampled, y_test_downsamples = resample_data(X_test, y_test, method='downsample')

    # Save data
    print("Saving downsampled data...")
    np.savetxt('XGB/data/train_data_downsampled.csv', np.c_[X_train_downsampled, y_train_downsamples], delimiter=',')
    np.savetxt('XGB/data/test_data_downsampled.csv', np.c_[X_test_downsampled, y_test_downsamples], delimiter=',')

    # Upsample
    print("Upsampling...")
    X_train_upsampled, y_train_upsampled = resample_data(X_train, y_train, method='upsample')
    X_test_upsampled, y_test_upsampled = resample_data(X_test, y_test, method='upsample')

    # Save data
    print("Saving upsampled data...")   
    np.savetxt('XGB/data/train_data_upsampled.csv', np.c_[X_train_upsampled, y_train_upsampled], delimiter=',')
    np.savetxt('XGB/data/test_data_upsampled.csv', np.c_[X_test_upsampled, y_test_upsampled], delimiter=',')

    print("Done!")
    return


if __name__ == '__main__':
    main()  