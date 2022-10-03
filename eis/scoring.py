import logging

from eis.EquivalentCircuitModels import EquivalentCircuitModel
from eis.EISDataIO import ECM_from_raw_strings
import numpy as np
from typing import Any, List, Optional
from copy import deepcopy
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from multiprocessing import Pool


def ECM_class_classification_scorer(
    circuit_labels_actual,
    circuit_labels_prediction
) -> float:
    encoder = LabelEncoder()
    actual = encoder.fit_transform(circuit_labels_actual)
    prediction = encoder.transform(circuit_labels_prediction)
    return balanced_accuracy_score(actual, prediction)


def _score_a_circuit(circuit, eis) -> float:
    name, parameters = circuit
    freq, Z = eis
    if pd.isna(parameters):
        return np.nan
    ecm = ECM_from_raw_strings(name, parameters)
    return circuit_CV_optimized_score(ecm, freq, Z)


def ECM_params_initial_guess_scorer(
    actual_EISata: pd.DataFrame, predicted_circuits_params: pd.DataFrame, n_cpus: Optional[int]
) -> float:
    pool = Pool(processes=n_cpus)

    circuits_and_data = zip(
        [(t.Circuit, t.Parameters) for t in predicted_circuits_params.itertuples()],
        [(t.freq, t.Z) for t in actual_EISata.itertuples()],
    )
    cv_scores = pool.starmap(_score_a_circuit, circuits_and_data)
    cv_scores = np.array(cv_scores)
    cv_scores = cv_scores[~np.isnan(cv_scores)]

    if cv_scores.shape[0] == 0:
        return np.nan
    else:
        return cv_scores.mean()


def circuit_rmse_score(
    circuit: EquivalentCircuitModel,
    frequencies: np.ndarray[Any, np.dtype[np.float_]],
    impedances: np.ndarray[Any, np.dtype[np.complex_]],
) -> float:
    """ A basic RMSE score of simulated impedance accuracies if you want that

    Args:
        circuit (EquivalentCircuitModel): the circuit we're scoring
        frequencies (np.ndarray[Any, np.dtype[np.float_]]): actual frequency data
        impedances (np.ndarray[Any, np.dtype[np.complex_]]): actual impedance data

    Returns:
        float: the absolute value of the RMSE of the impedances
    """
    simulated_impedances = circuit.simulate(frequencies)
    return np.abs(np.sqrt(((impedances - simulated_impedances) ** 2).mean()))


def circuit_CV_optimized_score(
    circuit: EquivalentCircuitModel,
    frequencies: np.ndarray[Any, np.dtype[np.float_]],
    impedances: np.ndarray[Any, np.dtype[np.complex_]],
    cv_folds: int = 3,
    max_penalty: float = 1e3,
) -> float:
    """ Offical competition scoring metric. Used to evaluate how good a set of initial parameter guesses for a circuit
    are. Does this by taking Cross Validation folds of the frequency and impedance data before the circuit is fit, then
    scoring the circuit on the data points which were not used in the fitting process.
    
    Final score is the absolute value of the complex RMSE of the out-of-bag data.

    Competition scoring will be done using default arguments.

    Args:
        circuit (EquivalentCircuitModel): the circuit with initial parameter guesses we're scoring
        frequencies (np.ndarray[Any, np.dtype[np.float_]]): actual frequency data
        impedances (np.ndarray[Any, np.dtype[np.complex_]]): actual impedance data
        max_penalty (float, optional): error applied in case circuit guess fit does not converge. Defaults to 1e3.

    Returns:
        float:  the absolute value of the RMSE of the out-of-bag impedances
    """
    # taking deepcopies to remove chance of a bug when run repeatedly.
    #   sneaky behavior - numpy arrays do not respect function scope in the same way as normal python variables
    og_circuit = deepcopy(circuit)
    frequencies = deepcopy(frequencies)
    impedances = deepcopy(impedances)

    # run cross-validation
    cv_scores = np.array([], dtype=np.complex_)
    fold_indices = np.arange(frequencies.shape[0])
    # np.random.shuffle(fold_indices)
    out_of_cv_fold_indices = np.array_split(fold_indices, cv_folds)
    for out_of_bag in out_of_cv_fold_indices:
        # Define the fitting data for current fold
        cv_mask = np.ones(frequencies.shape[0], bool)
        cv_mask[out_of_bag] = False
        frequencies_cv = frequencies[cv_mask]
        impedances_cv = impedances[cv_mask]

        # fit the circuit
        try:
            new_circuit = deepcopy(og_circuit)
            new_circuit.fit(frequencies_cv, impedances_cv)
        except RuntimeError as re:
            err_text = repr(re)
            if "Optimal parameters not found" in err_text:
                cv_scores = np.concatenate((cv_scores, np.array([complex(max_penalty)])))
                logging.debug("A CV fit was unable to converge")
            else:
                raise re
        except ValueError as ve:
            err_text = repr(ve)
            if "is not within the trust region." in err_text:
                cv_scores = np.concatenate((cv_scores, np.array([complex(max_penalty)])))
                logging.debug("A CV fit was unable to converge")
            else:
                raise ValueError(
                    f"""Invalid arguments when fiting circuit {circuit} with parameters: 
                    {zip(circuit.param_names, circuit.param_values)}. 
                    Check if your parameter guesses are within resonable bounds.
                    
                    Error Text:
                    {err_text}
                    """
                )

        # score the circuit
        else:
            simulated_impedances = new_circuit.simulate(frequencies)
            diff = (impedances[out_of_bag] - simulated_impedances[out_of_bag]) ** 2
            cv_scores = np.concatenate((cv_scores, diff))
    return np.abs(np.sqrt(cv_scores.mean()))


if __name__ == "__main__":
    from argparse import ArgumentParser
    from eis.EISDataIO import eis_dataframe_from_csv
    from os import cpu_count
    from sys import stderr
    import json

    parser = ArgumentParser(description="Submission Scoring CLI. Takes submission and test data and returns metrics.")
    parser.add_argument(
        "submission_file", type=str, help="Path to submission file CSV for scoring",
    )
    parser.add_argument(
        "answers_file", type=str, help="Path to answer data file CSV for scoring",
    )
    parser.add_argument(
        "scoring_mode", type=str, help="Scoring mode", choices=["circuits", "parameters", "both"],
    )
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=None,
        required=False,
        help="How many cpus to use for scoring. Defaults to all of them.",
    )
    args = parser.parse_args()

    if args.n_cpus is not None and args.n_cpus > cpu_count():
        raise ValueError(f"can't ask for a more cpus than you have, and you only have {cpu_count()}")

    submission = eis_dataframe_from_csv(args.submission_file)
    answers = eis_dataframe_from_csv(args.answers_file)

    if args.scoring_mode == "circuits":
        accuracy = ECM_class_classification_scorer(answers["Circuit"], submission["Circuit"])
        print(json.dumps({"accuracy": accuracy}))
        exit

    if args.scoring_mode == "parameters":
        print("Fitting parameters may take a minute, be patient...", file=stderr, flush=True)
        param_fit_loss = ECM_params_initial_guess_scorer(answers, submission, args.n_cpus)
        print(json.dumps({"param_fit_loss": param_fit_loss}))
        exit

    if args.scoring_mode == "both":
        print("Fitting parameters may take a minute, be patient...", file=stderr, flush=True)
        accuracy = ECM_class_classification_scorer(answers["Circuit"], submission["Circuit"])
        param_fit_loss = ECM_params_initial_guess_scorer(answers, submission, args.n_cpus)
        print(json.dumps({"accuracy": accuracy, "param_fit_loss": param_fit_loss}))
        exit

