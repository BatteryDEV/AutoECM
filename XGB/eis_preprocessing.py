# This script preproceses the spectra into a dataframe that susequenlt can be used for the machine learning approaches.
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def eis_dataframe_from_csv(csv_path) -> pd.DataFrame:
    """ Reads a CSV file of EIS data into a pandas dataframe
    Each row is an impedance meausurement

    Args:
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

# double of the function above 
def process_batch_element_params_str(Parameters):
    """Extracts the parameters of the Equivalent Circuit Model from the Parameters input"""
    Params = parse_circuit_params_from_str(Parameters)
    return np.array(list(Params.keys()))

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