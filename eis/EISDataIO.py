#!/usr/bin/env python
# coding: utf-8
# author: Raymond Gasper

import ast
from operator import index
from typing import Dict, Any
import numpy as np
import pandas as pd
from functools import partial
from eis.EquivalentCircuitModels import (
    RCPE,
    RCPE_RCPE,
    RCPE_RCPE_RCPE,
    RCPE_RCPE_RCPE_RCPE,
    L_R_RCPE,
    L_R_RCPE_RCPE,
    L_R_RCPE_RCPE_RCPE,
    RC_RC_RCPE_RCPE,
    RS_WS,
    RC_G_G,
    EquivalentCircuitModel,
)


CIRCUIT_NAME_TO_CIRCUIT: dict[str, type[EquivalentCircuitModel]] = {
    "RCPE": RCPE,
    "RCPE-RCPE": RCPE_RCPE,
    "RCPE-RCPE-RCPE": RCPE_RCPE_RCPE,
    "RCPE-RCPE-RCPE-RCPE": RCPE_RCPE_RCPE_RCPE,
    "L-R-RCPE": L_R_RCPE,
    "L-R-RCPE-RCPE": L_R_RCPE_RCPE,
    "L-R-RCPE-RCPE-RCPE": L_R_RCPE_RCPE_RCPE,
    "RC-RC-RCPE-RCPE": RC_RC_RCPE_RCPE,
    "Rs_Ws": RS_WS,
    "RC-G-G": RC_G_G,
}


def register_circuit_class(
    circuitClass: type[EquivalentCircuitModel], circuit_name: str
):
    circuit_name = circuit_name.strip()  # ignore any accidental whitespace
    if circuit_name in CIRCUIT_NAME_TO_CIRCUIT.keys():
        raise ValueError(f"Circuit name '{circuit_name}' is already defined")
    if circuitClass in CIRCUIT_NAME_TO_CIRCUIT.values():
        _reverse = {val: key for key, val in CIRCUIT_NAME_TO_CIRCUIT.items()}
        existing_key = _reverse[circuitClass]
        raise ValueError(
            f"Circuit class '{circuitClass.__name__}' is already registered under name '{existing_key}'"
        )

    CIRCUIT_NAME_TO_CIRCUIT[circuit_name] = circuitClass


def circuit_class_from_str(circuit_name: str) -> type[EquivalentCircuitModel]:
    name = circuit_name.strip()  # ignore any accidental whitespace
    return CIRCUIT_NAME_TO_CIRCUIT[name]


def parse_circuit_params_from_str(params_str: str) -> Dict[str, float]:
    return {
        item.split(":")[0].strip(): float(item.split(":")[1].strip())
        for item in params_str.split(",")
    }


def circuit_params_dict_to_str(params: Dict[str, float]) -> str:
    return ", ".join([f"{key}: {value}" for key, value in params.items()])


def ECM_from_raw_strings(circuit: str, params: str) -> EquivalentCircuitModel:
    circuit_class = circuit_class_from_str(circuit)
    params_dict = parse_circuit_params_from_str(params)
    return circuit_class(**params_dict)


def str2realarray(arr_str: str):
    try:
        arr_str = ",".join(arr_str.replace("[ ", "[").split())
        return np.array(ast.literal_eval(arr_str))
    except AttributeError:  # arr_str wasn't actually able to be interpreted as a string
        return pd.NA


def str2complexarray(arr_str: str):
    try:
        arr_str = arr_str.replace("\n", " ").replace("j ", "j,")
        return np.array(ast.literal_eval(arr_str))
    except AttributeError:  # arr_str wasn't actually able to be interpreted as a string
        return pd.NA


def dataframe_from_eis_data_zip(zipped_df_path: str) -> pd.DataFrame:
    """Reads the zip file of EIS data into a pandas dataframe

    Args:
        zipped_df_path (str): path to zipfile of EIS data

    Returns:
        pd.DataFrame: Dataframe with two or three columns - each row represents an EIS spectrum
            - freq  : short for frequency, column of real-number numpy arrays
            - Z     : short for impedance, column of imaginary-number numpy arrays
            - Circut: Equivalent Circuit Model labels assigned to the spectra (Optional)
    """
    df = pd.read_csv(zipped_df_path)
    df["freq"] = df["freq"].apply(str2realarray)
    df["Z"] = df["Z"].apply(str2complexarray)
    return df


def eis_dataframe_from_csv(csv_path) -> pd.DataFrame:
    """Reads a CSV file of EIS data into a pandas dataframe

    Args:
        csv_df_path (File-Like-Object): path to file of, or buffer of, EIS data in CSV format

    Returns:
        pd.DataFrame: Dataframe with two or three columns - each row represents an EIS spectrum
            - freq      : short for frequency, column of real-number numpy arrays
            - Z         : short for impedance, column of imaginary-number numpy arrays
            - Circuit   : Equivalent Circuit Model labels assigned to the spectra (Optional)
            - Parameters: Parameters of the Equivalent Circuit Models (Optional)
    """
    df = pd.read_csv(csv_path, index_col=0)

    def real2array(arraystr: str) -> np.ndarray:
        return np.array([float(c.strip("[]")) for c in arraystr.split(", ")])

    def comp2array(arraystr: str) -> np.ndarray:
        return np.array(
            [complex(c.strip("[]").replace(" ", "")) for c in arraystr.split(", ")]
        )

    if "freq" in df.columns:
        df["freq"] = df["freq"].apply(real2array)
    if "Z" in df.columns:
        df["Z"] = df["Z"].apply(comp2array)

    return df


def eis_dataframe_to_csv(df: pd.DataFrame, path) -> None:
    """Writes a pandas dataframe of EIS data into a CSV with the expected format

    Args:
        df (pd.DataFrame): DataFrame of EIS data. Frequence and Impedance columns optional.
            - Circuit   : Equivalent Circuit Model labels assigned to the spectra
            - Parameters: Parameters of the Equivalent Circuit Models
            - freq      : short for frequency, column of real-number numpy arrays (Optional)
            - Z         : short for impedance, column of imaginary-number numpy arrays (Optional)
        path (File-Like-Object) path of file, or buffer, to write data to in CSV format
    """
    dfc = df.copy()  # prevent bugs from modifying the df
    a2s = partial(np.array2string, max_line_width=10000, separator=", ")
    if "freq" in dfc.columns:
        dfc["freq"] = dfc["freq"].apply(a2s)
    if "Z" in dfc.columns:
        dfc["Z"] = dfc["Z"].apply(a2s)
    dfc.to_csv(path, index=True, sep=",")
    return
