# This script preproceses the spectra into a dataframe that susequenlt can be used for the machine learning approaches.
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import json 
from sklearn.preprocessing import LabelEncoder


def eis_dataframe_from_csv(csv_path) -> pd.DataFrame:
    """ Reads a CSV file of EIS data into a pandas dataframe
    Each row is an impedance meausurement

    Parameters:
    -----------
        csv_df_path (File-Like-Object): path to file of, or buffer of, EIS data in CSV format

    Return
    ------
    df : pandas.DataFrame
        A dataframe with two to four columns - each row represents an EIS spectrum
            - id: unique identifier for each measurement
            - freq: frequency of measurement
            - Z: short for impedance, column of imaginary-number numpy arrays
            - zimag: imaginary part of impedance
            - Circuit: Equivalent Circuit Model labels assigned to the spectra (Optional)
            - Parameters: Parameters of the Equivalent Circuit Models (Optional) 
    """
    df = pd.read_csv(csv_path, index_col=0)

    def real2array(arraystr: str):
        return np.array([float(c.strip("[]")) for c in arraystr.split(", ")])

    def comp2array(arraystr: str):
        return np.array(
            [complex(c.strip("[]").replace(" ", "")) for c in arraystr.split(", ")]
        )

    if "freq" in df.columns:
        df["freq"] = df["freq"].apply(real2array)
    if "Z" in df.columns:
        df["Z"] = df["Z"].apply(comp2array)

    # Rename that class Rs_Ws to R-Ws
    # df['Circuit'].loc[df['Circuit']=='Rs_Ws'] = 'R-Ws'

    return df

def process_batch_element_f(interpolated_basis):
    return interpolated_basis

def process_batch_element_zreal(freq, Z, interpolated_basis):
    """Interpolates the real part of the impedance onto a common frequency basis"""
    x = np.real(Z)
    f = interp1d(freq, x, fill_value="extrapolate")  # extrapolate to prevent errors
    return f(interpolated_basis)

def process_batch_element_zimag(freq, Z, interpolated_basis):
    """Interpolates the imaginary part of the impedance onto a common frequency basis"""
    x = np.imag(Z)
    f = interp1d(freq, x, fill_value="extrapolate")  # extrapolate to prevent errors
    return f(interpolated_basis)

def parse_circuit_params_from_str(params_str: str):
    """Split the parameters of the Equivalent Circuit Model into a dictionary"""
    return {
        item.split(":")[0].strip(): float(item.split(":")[1].strip())
        for item in params_str.split(",")
    }

def process_batch_element_params(Parameters):
    """Extracts the parameters of the Equivalent Circuit Model from the Parameters input"""
    Params = parse_circuit_params_from_str(Parameters)
    return np.array(list(Params.values()))

def unwrap_df(df):
    """Unwraps the frequency column into separate rows for each frequency to use UMAP"""
    df2 = pd.DataFrame(columns=["id", "freq", "zreal", "zimag"])
    for i in np.arange(df.shape[0]):
        f, zreal, zimag = df[["f", "zreal", "zimag"]].loc[i]
        idx = np.tile(i, f.size)
        df_ = pd.DataFrame(
            data=(idx, np.log(f), zreal, zimag), index=["id", "freq", "zreal", "zimag"]
        ).T
        df2 = df2.append(df_, ignore_index=True)
    return df2

def preprocess_data(file_name):
    """Preprocesses the data from the CSV filename into a dataframe"""
    ## Load Training Data
    df = eis_dataframe_from_csv(file_name)

    ## Interpolate onto the largest frequency union to prevent data leakage
    # f_max = df.freq.apply(np.max).min()
    # f_min = df.freq.apply(np.min).max()
    # For the training data f_max is 100.000 and f_min is 10.
    # Fixed in here to avoid issed in case this range is larger for the test dataset.
    # If the range is smaller for hte test datset retraining with relevant data and freq. range is necessary.
    interpolated_basis = np.geomspace(10, 1e5, num=30)

    df["f"] = df.apply(lambda x: process_batch_element_f(interpolated_basis), axis=1)
    df["zreal"] = df.apply(
        lambda x: process_batch_element_zreal(x.freq, x.Z, interpolated_basis), axis=1
    )
    df["zimag"] = df.apply(
        lambda x: process_batch_element_zimag(x.freq, x.Z, interpolated_basis), axis=1
    )
    return df

def exclude_outlier(df_train, df_test, z_mean_std_threshold=[8, 8], z_max_std_threshold=[10, 10], exclude_max=True):
    """Define limits to exclude the outlier from the training set, apply the limits to the test set
    This procedure was not used in the final version of the paper, but is included for completeness.

    Parameters
    ----------
    df_train: pandas.DataFrame
        Training data
    df_test: pandas.DataFrame   
        Test data
    z_mean_std_threshold: list
        Threshold for the standard deviation of the mean of the real and imaginary part of the impedance
    z_max_std_threshold: list
        Threshold for the standard deviation of the maximum of the real and imaginary part of the impedance
    exclude_max: bool
        If True, the standard deviation of the maximum of the real and imaginary part of the impedance is used to exclude outliers
    
    Return
    ------
    df_train: pandas.DataFrame
        Training data without outliers
    df_test: pandas.DataFrame
        Test data without outliers
    """
    real_mean_train = [df_train["zreal"][i].mean() for i in range(len(df_train))]
    imag_means_train = [df_train["zimag"][i].mean() for i in range(len(df_train))]
    real_max_val = [np.abs(df_train["zreal"][i]).max() for i in range(len(df_train))]
    imag_max_val = [np.abs(df_train["zimag"][i]).max() for i in range(len(df_train))]

    real_mean_test = [df_test["zreal"][i].mean() for i in range(len(df_test))]
    imag_means_test = [df_test["zimag"][i].mean() for i in range(len(df_test))]
    real_max_val_test = [np.abs(df_test["zreal"][i]).max() for i in range(len(df_test))]
    imag_max_val_test = [np.abs(df_test["zimag"][i]).max() for i in range(len(df_test))]

    # Define thresholds as 6 standard deviations from the mean based on the training set
    real_mean_threshold = np.mean(real_mean_train) + z_mean_std_threshold[0] * np.std(real_mean_train)
    imag_mean_threshold = np.mean(imag_means_train) + z_mean_std_threshold[1] * np.std(imag_means_train)
    if exclude_max:
        real_max_val_threshold = np.mean(real_max_val) + z_max_std_threshold[0] * np.std(real_max_val)
        imag_max_val_threshold = np.mean(imag_max_val) + z_max_std_threshold[0] * np.std(imag_max_val)
    else:
        real_max_val_threshold = np.Inf
        imag_max_val_threshold = np.Inf

    # Get indices of the outliers
    outliers_train = []
    outliers_train.extend(np.where(np.array(real_mean_train) > real_mean_threshold)[0])
    outliers_train.extend(np.where(np.array(imag_means_train) > imag_mean_threshold)[0])
    outliers_train.extend(np.where(np.array(real_max_val) > real_max_val_threshold)[0])
    outliers_train.extend(np.where(np.array(imag_max_val) > imag_max_val_threshold)[0])
    # Get the unique indices
    outliers_train = np.unique(outliers_train)
    print(outliers_train)
    
    # Get the indices of the outliers in the test set
    outliers_test = []
    outliers_test.extend(np.where(np.array(real_mean_test) > real_mean_threshold)[0])
    outliers_test.extend(np.where(np.array(imag_means_test) > imag_mean_threshold)[0])
    outliers_test.extend(np.where(np.array(real_max_val_test) > real_max_val_threshold)[0])
    outliers_test.extend(np.where(np.array(imag_max_val_test) > imag_max_val_threshold)[0])
    # Get the unique indices
    outliers_test = np.unique(outliers_test)
    print(outliers_test)

    # Exclude the outlier
    print(f'Train data: {len(df_train)} spectra')
    df_train_outlier_excluded = df_train.copy()
    df_train_outlier_excluded = df_train_outlier_excluded.drop(outliers_train).reset_index(drop=True)
    print(f'Train data: {len(df_train_outlier_excluded)} spectra after outliers removed')

    print(f'Test data: {len(df_test)} spectra')
    df_test_outlier_excluded = df_test.copy()
    df_test_outlier_excluded = df_test_outlier_excluded.drop(outliers_test).reset_index(drop=True)
    print(f'Test data: {len(df_test_outlier_excluded)} spectra after outliers removed')
    return df_train_outlier_excluded, df_test_outlier_excluded

def plot_all_spectra(df_sorted, fig=None, ax=None, plot_real=True, save=0, color='k', return_fig=False):
    """Plot all the spectra in the dataframe"""
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(len(df_sorted)):
        if plot_real:
            ax.plot(df_sorted["f"][i], df_sorted["zreal"][i], color=color, alpha=0.6)
        else:
            ax.plot(df_sorted["f"][i], df_sorted["zimag"][i], color=color, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    if plot_real:
        ax.set_ylabel("Real(Z)")
    else:
        ax.set_ylabel("Imag(Z)")
    ax.set_title("All Spectra")
    if save:
        fig.savefig("figures/all_spectra.png", dpi=300)

    if return_fig:
        return fig, ax
    else:
        plt.show()

def eis_label_encoder(le_f = 'models/labels.json'):
    """Load the label encoder"""
    with open(le_f, 'r') as f:
        mapping = json.load(f)
        le = LabelEncoder()

    nb_classes = len(mapping.keys())
    mapping['classes'] = [mapping[str(int(i))] for i in range(nb_classes)]
    le.classes_ = np.array(mapping['classes'])

    return le, mapping
