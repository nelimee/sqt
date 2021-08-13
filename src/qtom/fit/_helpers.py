import typing as ty

import numpy
from qiskit import QuantumCircuit
from qiskit.result import Result, marginal_counts

from qtom import _constants
from qtom.basis import BaseMeasurementBasis, PauliMeasurementBasis


def compute_frequencies(
    result: Result,
    tomographied_circuit: QuantumCircuit,
    qubit_index: ty.Optional[int] = None,
    is_parallel: bool = False,
    basis: ty.Optional[BaseMeasurementBasis] = None,
) -> ty.Dict[str, ty.Dict[str, float]]:
    """Compute an approximation of each expectation value with the frequency.

    This function takes as input the results of a given job, the circuit that
    have been tomographied (without the basis change) and returns the observed
    frequency for each measurement.

    :param result: the data returned by IBM chips.
    :param tomographied_circuit: the quantum circuit we are interested in.
        Should be the QuantumCircuit instance **before** appending the
        tomography basis change. In other words, the name of this
        QuantumCircuit should not include the tomography basis identifier.
    :param qubit_index: index of the qubit the tomography process has been
        applied on.
    :param is_parallel: if True, several tomography experiment have been
        performed in parallel and the results are including the raw
        measurements. If this flag is true, a call to marginal_counts is
        performed with the given qubit_index in order to get the right
        measurements.
    :param basis: the tomography basis used. Default to PauliMeasurementBasis().

    :return: the estimated frequencies as a mapping
        {basis_change_str -> {state -> frequency}} where basis_change_str is
        the name of the quantum circuit performing the basis change, state is
        either "0" or "1" for 1-qubit and frequency is the estimated frequency.
    """
    if basis is None:
        basis = PauliMeasurementBasis()
    # Compute the probabilities
    frequencies: ty.Dict[str, ty.Dict[str, float]] = dict()
    tc_name: str = tomographied_circuit.name
    qubit_index_str: str = ""
    if qubit_index is not None and not is_parallel:
        qubit_index_str = f"_{qubit_index}"

    for basis_change in map(lambda qc: qc.name, basis.basis_change_circuits()):
        counts = result.get_counts(f"{tc_name}_{basis_change}{qubit_index_str}")
        # If there is a need to marginalise the counts because the circuits were
        # executed in parallel, do it now!
        if is_parallel and qubit_index is not None:
            counts = marginal_counts(counts, indices=[qubit_index])
        frequencies[basis_change] = counts
        # Change the counts in probabilities
        count = sum(frequencies[basis_change].values())
        for state in frequencies[basis_change]:
            frequencies[basis_change][state] /= count
    return frequencies


def _couter(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
    """Complex conjugate outer product."""
    # TODO: was numpy.outer(a.T.conj(), b), might be a mistake
    return numpy.outer(a, b.T.conj())


def _bloch_vector_to_density_matrix(s: numpy.ndarray) -> numpy.ndarray:
    """
    Take a 3-dimensional Bloch vector and returns the corresponding density matrix.

    :param s: A 3-dimensional real vector representing a point within the Bloch
        sphere.
    :return: a 2 by 2 density matrix corresponding to the given state.
    """
    return (
        _constants.I + s[0] * _constants.X + s[1] * _constants.Y + s[2] * _constants.Z
    ) / 2
