#!/usr/bin/env python
# coding: utf-8

from eis.EquivalentCircuitModels import EquivalentCircuitModel
from typing import Optional, Any
import numpy as np
import matplotlib.pyplot as plt


def plot_eis(
    frequencies: np.ndarray[Any, np.dtype[np.float_]],
    impedance: np.ndarray[Any, np.dtype[np.complex_]],
    title: Optional[str] = None,
    ECM: Optional[EquivalentCircuitModel] = None,
):
    """ Creates a single figure w/ both Bode and Nyquist plots of a single EIS spectrum.
    Plots the results of a simulated circuit as well if provided

    Args:
        frequency (np.ndarray[Any, np.dtype[np.float_]]): numpy array of frequency values. Real, positive numbers
        impedance (np.ndarray[Any, np.dtype[np.complex_]]): numpy array of impedance values. Imaginary numbers
        title (Optional[str]): A figure title. Defaults to None.
        ECM (Optional[Circuit]) A Equivalent Circuit Model. Defaults to None.
    """

    fig, ax = plt.subplots(1, 2)
    if ECM is not None:
        sim_impedance = ECM.simulate(frequencies)

    # Bode Plot
    ax2 = ax[0].twinx()
    ax[0].loglog(frequencies, np.abs(impedance), "ko")
    ax2.semilogx(frequencies, np.angle(impedance, deg=True), "go")
    ax[0].set_title("Bode")
    ax[0].set_xlabel("Freq [Hz]")
    ax[0].set_ylabel(r"|Z| [$\Omega$]", color="k")
    ax2.set_ylabel(r"$\angle$Z [$^\circ$]", color="g", rotation=270, labelpad=15)
    if ECM is not None:
        ax[0].loglog(frequencies, np.abs(sim_impedance), "k-")
        ax2.semilogx(frequencies, np.angle(sim_impedance, deg=True), "g-")

    # Nyquist Plot
    ax[1].plot(np.real(impedance), -np.imag(impedance), "bo")
    ax[1].set_aspect("equal")
    ax[1].set_title("Nyquist")
    ax[1].set_xlabel(r"Re(Z) [$\Omega$]")
    ax[1].set_ylabel(r"-Im(Z) [$\Omega$]")
    if ECM is not None:
        ax[1].plot(np.real(sim_impedance), -np.imag(sim_impedance), "b-")

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    plt.yticks(color="g")
    plt.show()


if __name__ == "__main__":
    ...
