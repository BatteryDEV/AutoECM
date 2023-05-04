# This script preproceses the spectra into a dataframe that susequenlt can be used for the machine learning approaches.
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import json
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
from utils import visualize_raw_spectra


def eis_dataframe_from_csv(csv_path) -> pd.DataFrame:
    """Reads a CSV file of EIS data into a pandas dataframe
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


def process_batch_element_params_str(Parameters):
    Params = parse_circuit_params_from_str(Parameters)
    return np.array(list(Params.keys()))


def unwrap_df(df: pd.DataFrame) -> pd.DataFrame:
    """Unwraps the frequency column into separate rows for each frequency to use UMAP"""
    df2 = pd.DataFrame(columns=["id", "freq", "zreal", "zimag"])
    for ind in df.index:
        f, zreal, zimag = df[["f", "zreal", "zimag"]].loc[ind]
        idx = np.tile(ind, f.size)
        df_ = pd.DataFrame(
            data=(idx, np.log(f), zreal, zimag), index=["id", "freq", "zreal", "zimag"]
        ).T
        df2 = df2.append(df_, ignore_index=True)
    return df2


def preprocess_data(file_name: str, num_points: int = 30) -> pd.DataFrame:
    """Preprocesses the data from the CSV filename into a dataframe"""
    # Load Training Data
    df = eis_dataframe_from_csv(file_name)

    # Interpolate onto the largest frequency union to prevent data leakage
    # f_max = df.freq.apply(np.max).min()
    # f_min = df.freq.apply(np.min).max()
    # For the training data f_max is 100.000 and f_min is 10.
    # Fixed in here to avoid issed in case this range is larger for the test dataset.
    # If the range is smaller for hte test datset retraining with relevant data and freq. range is necessary.
    interpolated_basis = np.geomspace(10, 1e5, num=num_points)

    df["f"] = df.apply(lambda x: process_batch_element_f(interpolated_basis), axis=1)
    df["zreal"] = df.apply(
        lambda x: process_batch_element_zreal(x.freq, x.Z, interpolated_basis), axis=1
    )
    df["zimag"] = df.apply(
        lambda x: process_batch_element_zimag(x.freq, x.Z, interpolated_basis), axis=1
    )
    return df


def get_stats_zreal(df, variable="z_real_mean"):
    mean = np.mean(df[variable])
    std = np.std(df[variable])
    return mean, std


def filter_df(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame = None,
    mono_thres=1e-3,
    plot_outliers: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Excludes outliers from the training data and test data if provided
    The criteria are described in more detail in the associated paper.

    Parameters
    ----------
    df_train: pd.DataFrame
        Training data
    df_test: pd.DataFrame
        Test data (Optional)
    mono_thres: float
        Threshold for the monotonicity of the real part of the impedance (Default: 1e-3)
    plot_outliers: bool
        Plot the outliers (Default: False)

    Returns
    -------
    df_train: pd.DataFrame
        Filtered training data
    df_test: pd.DataFrame
        Filtered rest data (if provided)
    """

    def is_monotonic(x, thres=mono_thres):
        return np.all(np.diff(x) <= thres)

    def filter_single_df(df):
        index_train = df.index
        # Rewrite this function!
        f_, Z_, index_q1 = filter_simple_qc_list(df, neg_real_z_filter=True)
        index_list_q1 = [i for i in index_train if i in index_q1]

        df_q1 = df.iloc[index_list_q1].copy()
        z_mono = df_q1["zreal"].apply(lambda x: is_monotonic(x))

        # create a new dataframe with only the spectra that are monotonic
        df_q1_mono = df_q1[z_mono == 1].copy()

        if plot_outliers:
            # get the indices of the spectra that are not monotonic
            index_list_nmono = df_q1[z_mono == 0].index
            index_list_filtered = [
                i for i in index_train if i not in index_q1 and i not in index_list_nmono
            ]

            fig = visualize_raw_spectra(
                df.loc[index_list_filtered].copy(),
                show=1,
                save_figs=0,
                row_col_ratio=0.6,
                pdf=True,
                fig_name="NQ1_filtered_spectra",
                sup_title="Spectra are not in the first quadrant and did not meet simple QCs, labeled data",
                axis_off=False,
            )
            fig.show()
        return df_q1_mono

    df_train_filtered = filter_single_df(df_train)
    print(f"Train data: {len(df_train_filtered)} spectra after outliers removed")

    if df_test is not None:
        df_test_filtered = filter_single_df(df_test)
        print(f"Test data: {len(df_test_filtered)} spectra after outliers removed")
        return df_train_filtered, df_test_filtered
    else:
        return df_train_filtered


def eis_label_encoder(
    le_f: str = "data/le_name_mapping.json",
) -> Tuple[LabelEncoder, dict]:
    """Load the label encoder"""
    with open(le_f, "r") as f:
        mapping = json.load(f)
        le = LabelEncoder()

    nb_classes = len(mapping.keys())
    mapping["classes"] = [mapping[str(int(i))] for i in range(nb_classes)]
    le.classes_ = np.array(mapping["classes"])

    return le, mapping


def unwrap_z(df):
    df["zreal"] = df.apply(lambda x: np.real(x.Z), axis=1)
    df["zimag"] = df.apply(lambda x: np.imag(x.Z), axis=1)
    return df


def sort_circuits(df):
    # Sort the data frame by circuit.
    df = df.sort_values(by=["Circuit"])
    return df


def interpolate_to_freq_range(df, interpolated_basis):
    # Interpolate onto the largest frequency union to prevent data leakage

    df["f"] = df.apply(lambda x: process_batch_element_f(interpolated_basis), axis=1)
    df["zreal"] = df.apply(
        lambda x: process_batch_element_zreal(x.freq, x.Z, interpolated_basis), axis=1
    )
    df["zimag"] = df.apply(
        lambda x: process_batch_element_zimag(x.freq, x.Z, interpolated_basis), axis=1
    )
    return df


def filter_pos_imag(frequencies, Z, neg_real_z_filter=True):
    """
    Strategies for filtering out data that is not in the first quadrant.
    Based on ignoreBelowX function from impedancepy with modifications.
    """
    range_pos = np.abs(np.max(-np.imag(Z)))
    range_neg = np.abs(np.min(-np.imag(Z)))
    if range_neg > 2 * range_pos:
        # if the range of the positive imaginary part is less than half the range of the negative imaginary part
        # then we can assume that the data is not in the first quadrant
        # and we can filter out the data
        Z = []
        frequencies = []
    elif neg_real_z_filter:
        # Check whether the real part of the impedance is negative
        # If so, filter out.
        if np.any(np.real(Z) < 0):
            Z = []
            frequencies = []
    else:
        mask = np.imag(Z) < 0
        frequencies = frequencies[mask]
        Z = Z[mask]
    return frequencies, Z


def filter_simple_qc_list(
    df: pd.DataFrame,
    scale: float = 20e-3,
    verbose=False,
    neg_real_z_filter=False,
):
    """Convert a dataframe to a two lists f, Z that can be used in impedance.py"""
    f = df["freq"].values.tolist()
    Z = df["Z"].values.tolist()
    f_list = []
    Z_list = []
    index_all = df.index
    index_list = []
    for i in range(len(Z)):
        # keep only the impedance data in the first quandrant
        f_, Z_ = filter_pos_imag(f[i], Z[i], neg_real_z_filter=neg_real_z_filter)
        # check whether Z_ is empty
        if len(Z_) <= 10:
            if verbose:
                print(f"Empty Z_ at index {i}")
        else:
            f_list.append(f_)
            Z_list.append(Z_)
            index_list.append(index_all[i])
    return f_list, Z_list, index_list
