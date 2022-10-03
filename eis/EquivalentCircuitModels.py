from typing import Any, Callable, List, Tuple
import numpy as np
from scipy.optimize import curve_fit
from eis.utils import flatten2list


class Element:
    """ element of an equivalent circuit model """

    def __init__(self, **parameters) -> None:
        pass

    def simulate(
        self, frequencies: np.ndarray[Any, np.dtype[np.float_]]
    ):
        """ Generate the impedance spectrum from this element based on provided frequencies

        Args:
            frequencies (np.ndarray[Any, np.dtype[np.float_]])

        Returns:
            np.ndarray[Any, np.dtype[np.complex_]] | complex
        """
        ...

    @property
    def param_fitting_bounds(self) -> Tuple[List[float], List[float]]:
        """ Give the upper and bounds used when fitting parameters for this element

        Returns:
            List[float], List[float]: Lower Bounds, Upper Bounds. In order of element signature 
        """
        ...


class EquivalentCircuitModel:
    """ Equivalent Circuit Model with methods for simulation and parameter fitting. Slightly non-standard definitions of
    elements just to be an easy conversion from how QuantumScape defined circuits in their data. 
    
    If you want to fit a circuit, define it with your initial guesses for the parameters """

    circuit: List[Tuple[type[Element], List[str], List[float]]] = []
    "List of Tups:  Element, [param_names], [param_values]"

    def __init__(self, **parameters) -> None:
        ...

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, type(self)) and self.circuit == __o.circuit

    @property
    def _circuit(self) -> List[Element]:
        """ List of instantiated elements, lazily evaluated """
        return [meta[0](*meta[2]) for meta in self.circuit]

    @property
    def param_names(self) -> List[str]:
        return flatten2list([element[1] for element in self.circuit])

    @property
    def param_values(self) -> List[str]:
        return flatten2list([element[2] for element in self.circuit])

    @property
    def param_fitting_bounds(self) -> Tuple[List[float], List[float]]:
        return (
            flatten2list(_element.param_fitting_bounds[0] for _element in self._circuit),
            flatten2list(_element.param_fitting_bounds[1] for _element in self._circuit),
        )

    def reassign_parameters_via_args(self, *parameters):
        parameters_list = list(parameters)
        for index in range(len(self.circuit)):
            eletype, param_names, initial_param_values = self.circuit[index]
            num_params_this_element = len(param_names)
            # "pop" off slices of the optimized parameters going left-to-right
            optimized_params_this_element = parameters_list[:num_params_this_element]
            parameters_list = parameters_list[num_params_this_element:]
            self.circuit[index] = (eletype, param_names, list(optimized_params_this_element))

    def simulate(self, frequencies: np.ndarray[Any, np.dtype[np.float_]]) -> np.ndarray[Any, np.dtype[np.complex_]]:
        return np.sum(component.simulate(frequencies) for component in self._circuit)

    def _wrapped_sim_for_fit(self) -> Callable:
        def update_circuit_then_simulate(frequencies: np.ndarray[Any, np.dtype[np.float_]], *new_parameters):
            self.reassign_parameters_via_args(*new_parameters)
            simulated_impedances = self.simulate(frequencies)
            return np.hstack([simulated_impedances.real, simulated_impedances.imag])

        return update_circuit_then_simulate

    def fit(
        self, frequencies: np.ndarray[Any, np.dtype[np.float_]], impedances: np.ndarray[Any, np.dtype[np.complex_]]
    ):
        optimized_parameters, parameter_covariance = curve_fit(
            self._wrapped_sim_for_fit(),
            frequencies,
            np.hstack([impedances.real, impedances.imag]),
            bounds=self.param_fitting_bounds,
            p0=self.param_values,
        )
        self.reassign_parameters_via_args(*optimized_parameters)


class R(Element):
    def __init__(self, R: float):
        self.R = R

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, type(self)) and self.R == __o.R

    def simulate(self, frequencies: np.ndarray[Any, np.dtype[np.float_]]) -> complex:
        # freq is unused, but included to match the expected function signature of all the circuits
        return self.R

    @property
    def param_fitting_bounds(self) -> Tuple[List[float], List[float]]:
        return [0], [np.inf]


class L(Element):
    def __init__(self, L: float):
        self.L = L

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, type(self)) and self.L == __o.L

    def simulate(self, frequencies: np.ndarray[Any, np.dtype[np.float_]]) -> np.ndarray[Any, np.dtype[np.complex_]]:
        return (1j * frequencies) * self.L

    @property
    def param_fitting_bounds(self) -> Tuple[List[float], List[float]]:
        return [0], [np.inf]


class RC(Element):
    def __init__(self, R: float, C: float):
        self.R = R
        self.C = C

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, type(self)) and self.R == __o.R and self.C == __o.C

    def simulate(self, frequencies: np.ndarray[Any, np.dtype[np.float_]]) -> np.ndarray[Any, np.dtype[np.complex_]]:
        return self.R / (1 + self.R * self.C * (1j * frequencies))

    @property
    def param_fitting_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0], [np.inf, np.inf]


class RCPE(Element):
    def __init__(self, R: float, CPE_C: float, CPE_t: float):
        self.R = R
        self.CPE_C = CPE_C
        self.CPE_t = CPE_t

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, type(self)) and self.R == __o.R and self.CPE_C == __o.CPE_C and self.CPE_t == __o.CPE_t

    def simulate(self, frequencies: np.ndarray[Any, np.dtype[np.float_]]) -> np.ndarray[Any, np.dtype[np.complex_]]:
        return self.R / (1 + self.R * self.CPE_C * (1j * frequencies) ** self.CPE_t)

    @property
    def param_fitting_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, 0], [np.inf, np.inf, 10]
        # NOTE sources suggest using 1 for CPE_t upper bound, but QS includes some slightly > 1 values!


class RW(Element):
    W_p_scale = 6.5  # Scaling factor?? Where did this come from again?? TODO confirm with QS staff

    def __init__(self, W_R: float, W_T: float, W_p: float):
        self.W_R = W_R
        self.W_T = W_T
        self.W_p = W_p / self.W_p_scale

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, type(self)) and self.W_R == __o.W_R and self.W_T == __o.W_T and self.W_p == __o.W_p

    def simulate(self, frequencies: np.ndarray[Any, np.dtype[np.float_]]) -> np.ndarray[Any, np.dtype[np.complex_]]:
        return (
            self.W_R * np.tanh((1j * frequencies * self.W_T) ** self.W_p) / ((1j * frequencies * self.W_T) ** self.W_p)
        )

    @property
    def param_fitting_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, 0], [np.inf, np.inf, 10 * self.W_p_scale]


class G(Element):
    def __init__(self, R: float, t: float) -> None:
        self.R = R
        self.t = t

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, type(self)) and self.R == __o.R and self.t == __o.t

    def simulate(self, frequencies: np.ndarray[Any, np.dtype[np.float_]]) -> np.ndarray[Any, np.dtype[np.complex_]]:
        return self.R / np.sqrt(1 + 1j * frequencies * self.t)

    @property
    def param_fitting_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0], [np.inf, np.inf]


class RC_G_G(EquivalentCircuitModel):
    def __init__(self, R1, C1, R_g1, t_g1, R_g2, t_g2) -> None:
        self.circuit = [
            (RC, ["R1", "C1"], [R1, C1]),
            (G, ["R_g1", "t_g1"], [R_g1, t_g1]),
            (G, ["R_g2", "t_g2"], [R_g2, t_g2]),
        ]


class RCPE_RCPE(EquivalentCircuitModel):
    def __init__(self, R1: float, R2: float, CPE1_t: float, CPE1_C: float, CPE2_t: float, CPE2_C: float):
        self.circuit = [
            (RCPE, ["R1", "CPE1_C", "CPE1_t"], [R1, CPE1_C, CPE1_t]),
            (RCPE, ["R2", "CPE2_C", "CPE2_t"], [R2, CPE2_C, CPE2_t]),
        ]


class RCPE_RCPE_RCPE(EquivalentCircuitModel):
    def __init__(
        self,
        R1: float,
        R2: float,
        R3: float,
        CPE1_t: float,
        CPE1_C: float,
        CPE2_t: float,
        CPE2_C: float,
        CPE3_t: float,
        CPE3_C: float,
    ):
        self.circuit = [
            (RCPE, ["R1", "CPE1_C", "CPE1_t"], [R1, CPE1_C, CPE1_t]),
            (RCPE, ["R2", "CPE2_C", "CPE2_t"], [R2, CPE2_C, CPE2_t]),
            (RCPE, ["R3", "CPE3_C", "CPE3_t"], [R3, CPE3_C, CPE3_t]),
        ]


class RCPE_RCPE_RCPE_RCPE(EquivalentCircuitModel):
    def __init__(
        self,
        R1: float,
        R2: float,
        R3: float,
        R4: float,
        CPE1_t: float,
        CPE1_C: float,
        CPE2_t: float,
        CPE2_C: float,
        CPE3_t: float,
        CPE3_C: float,
        CPE4_t: float,
        CPE4_C: float,
    ):
        self.circuit = [
            (RCPE, ["R1", "CPE1_C", "CPE1_t"], [R1, CPE1_C, CPE1_t]),
            (RCPE, ["R2", "CPE2_C", "CPE2_t"], [R2, CPE2_C, CPE2_t]),
            (RCPE, ["R3", "CPE3_C", "CPE3_t"], [R3, CPE3_C, CPE3_t]),
            (RCPE, ["R4", "CPE4_C", "CPE4_t"], [R4, CPE4_C, CPE4_t]),
        ]


class RC_RC_RCPE_RCPE(EquivalentCircuitModel):
    def __init__(
        self,
        R1: float,
        R2: float,
        R3: float,
        R4: float,
        C1: float,
        C2: float,
        CPE3_C: float,
        CPE4_C: float,
        CPE3_t: float,
        CPE4_t: float,
    ):
        self.circuit = [
            (RC, ["R1", "C1"], [R1, C1]),
            (RC, ["R2", "C2"], [R2, C2]),
            (RCPE, ["R3", "CPE3_C", "CPE3_t"], [R3, CPE3_C, CPE3_t]),
            (RCPE, ["R4", "CPE4_C", "CPE4_t"], [R4, CPE4_C, CPE4_t]),
        ]


class L_R_RCPE(EquivalentCircuitModel):
    def __init__(self, L1: float, R1: float, R2: float, CPE1_C: float, CPE1_t: float):
        self.circuit = [
            (L, ["L1"], [L1]),
            (R, ["R1"], [R1]),
            (RCPE, ["R2", "CPE1_C", "CPE1_t"], [R2, CPE1_C, CPE1_t]),
        ]


class L_R_RCPE_RCPE(EquivalentCircuitModel):
    def __init__(self, L1, R1: float, R2: float, CPE1_C: float, CPE1_t: float, R3: float, CPE2_C: float, CPE2_t: float):
        self.circuit = [
            (L, ["L1"], [L1]),
            (R, ["R1"], [R1]),
            (RCPE, ["R2", "CPE1_C", "CPE1_t"], [R2, CPE1_C, CPE1_t]),
            (RCPE, ["R3", "CPE2_C", "CPE2_t"], [R3, CPE2_C, CPE2_t]),
        ]


class L_R_RCPE_RCPE_RCPE(EquivalentCircuitModel):
    def __init__(
        self,
        L1,
        R1: float,
        R2: float,
        CPE1_C: float,
        CPE1_t: float,
        R3: float,
        CPE2_C: float,
        CPE2_t: float,
        R4: float,
        CPE3_C: float,
        CPE3_t: float,
    ):
        self.circuit = [
            (L, ["L1"], [L1]),
            (R, ["R1"], [R1]),
            (RCPE, ["R2", "CPE1_C", "CPE1_t"], [R2, CPE1_C, CPE1_t]),
            (RCPE, ["R3", "CPE2_C", "CPE2_t"], [R3, CPE2_C, CPE2_t]),
            (RCPE, ["R4", "CPE3_C", "CPE3_t"], [R4, CPE3_C, CPE3_t]),
        ]


class RS_WS(EquivalentCircuitModel):
    def __init__(self, R1: float, W1_R: float, W1_T: float, W1_p: float):
        self.circuit = [
            (R, ["R1"], [R1]),
            (RW, ["W1_R", "W1_T", "W1_p"], [W1_R, W1_T, W1_p]),
        ]
