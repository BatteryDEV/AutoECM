from .EISDataIO import (
    ECM_from_raw_strings,
    register_circuit_class,
    circuit_params_dict_to_str,
    parse_circuit_params_from_str,
)
from .EquivalentCircuitModels import EquivalentCircuitModel, RC


def test_circuit_parameters_order_doesnt_matter():
    circuit_string = "RC-RC-RCPE-RCPE"
    param_string = "R1: 8.69e-01, R2: 6.86e-02, R3: 1.34e+00, R4: 1.57e+05, C2: 4.35e-05, CPE3_C: 5.42e-04, CPE4_t: 9.50e-01, CPE4_C: 3.07e-01, C1: 2e-9, CPE3_t: 0.9"  # noqa: E501
    ecm1 = ECM_from_raw_strings(circuit_string, param_string)
    # swapped R1 and R2 positions
    param_string = "R2: 6.86e-02, R1: 8.69e-01, R3: 1.34e+00, R4: 1.57e+05, C2: 4.35e-05, CPE3_C: 5.42e-04, CPE4_t: 9.50e-01, CPE4_C: 3.07e-01, C1: 2e-9, CPE3_t: 0.9"  # noqa: E501
    ecm2 = ECM_from_raw_strings(circuit_string, param_string)
    assert ecm1 == ecm2


def test_circuit_params_writing_parsing():
    param_string_1 = "R1: 8.69e-01, R2: 6.86e-02, R3: 1.34e+00, R4: 1.57e+05, C2: 4.35e-05, CPE3_C: 5.42e-04, CPE4_t: 9.50e-01, CPE4_C: 3.07e-01, C1: 2e-9, CPE3_t: 0.9"  # noqa: E501
    params_dict_1 = parse_circuit_params_from_str(param_string_1)
    params_string_2 = circuit_params_dict_to_str(params_dict_1)
    params_dict_2 = parse_circuit_params_from_str(params_string_2)
    assert params_dict_2 == params_dict_1


def test_can_add_new_circuit_to_registry():
    new_circuit_str = "RC"
    param_str = "R1: 1, C1: 1"

    class justRC(EquivalentCircuitModel):
        def __init__(self, R1: float, C1: float) -> None:
            self.circuit = [
                (RC, ["R1", "C1"], [R1, C1]),
            ]

    register_circuit_class(justRC, new_circuit_str)
    ecm = ECM_from_raw_strings(new_circuit_str, param_str)
    assert isinstance(ecm, EquivalentCircuitModel)
