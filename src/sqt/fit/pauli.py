import typing as ty

import numpy
from qiskit import QuantumCircuit
from qiskit.result import Result

from sqt import _constants
from sqt.basis import BaseMeasurementBasis, PauliMeasurementBasis
from sqt.fit._helpers import compute_frequencies


def frequencies_to_pauli_reconstruction(
    frequencies: ty.Dict[str, ty.Dict[str, float]],
    basis: BaseMeasurementBasis,
) -> numpy.ndarray:
    """Compute the density matrix from the given frenquencies.

    This function use the fact that any density matrix can be decomposed as
    a weighted sum of Pauli matrices and finds the coefficients of this
    weighted sum.

    :param frequencies: a mapping {basis_change_string -> {state -> frequency}}
        that is used to reconstruct the density matrix. The possible values for
        basis_change_string are given by the measurement basis used. For
        1-qubit tomography experiment, state can be either "0" or "1".
    :param basis: the measurement basis used for tomography. Should be the
        Pauli basis.
    """
    assert (
        basis.name == "pauli"
    ), "Pauli reconstruction method is only implemented for Pauli measurements."
    assert (
        basis.qubit_number == 1
    ), "Only 1-qubit tomography is supported for the moment."
    return (
        _constants.I
        + (2 * frequencies["bcH"].get("0", 0) - 1) * _constants.X
        + (2 * frequencies["bcSdgH"].get("0", 0) - 1) * _constants.Y
        + (2 * frequencies["bcI"].get("0", 0) - 1) * _constants.Z
    ) / 2


def post_process_tomography_results_pauli(
    result: Result,
    tomographied_circuit: QuantumCircuit,
    qubit_index: ty.Optional[int] = None,
    is_parallel: bool = False,
    basis: ty.Optional[BaseMeasurementBasis] = None,
) -> numpy.ndarray:
    """
    Compute and return the density matrix computed via state tomography.

    :param result: the Result instance returned by the QPU after executing all
        the circuits returned by the one_qubit_tomography_circuits function.
    :param tomographied_circuit: the quantum circuit instance that is currently
        tomographied. Used to recover the circuit name.
    :param qubit_index: index of the qubit used to perform the tomography
        experiments. Internally, the function adapts the naming scheme according
        to this parameter.
    :return: the 2 by 2 density matrix representing the prepared quantum state.
    """
    if basis is None:
        basis = PauliMeasurementBasis()
    frequencies: ty.Dict[str, ty.Dict[str, float]] = compute_frequencies(
        result, tomographied_circuit, qubit_index, is_parallel, basis
    )

    return frequencies_to_pauli_reconstruction(frequencies, basis)
