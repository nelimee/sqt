import numpy
from qiskit import QuantumCircuit
from qiskit.result import Result

from sqt import _constants
from sqt.basis.base import BaseMeasurementBasis
from sqt.counts import Counts
from sqt.fit._helpers import compute_frequencies


def frequencies_to_pauli_reconstruction(
    frequencies: list[dict[str, Counts]],
    basis: BaseMeasurementBasis,
) -> list[numpy.ndarray]:
    """Compute the density matrix from the given frenquencies.

    This function use the fact that any density matrix can be decomposed as
    a weighted sum of Pauli matrices and finds the coefficients of this
    weighted sum.

    Args:
        frequencies: a list of mappings {basis_change_string -> {state
            -> frequency}} that is used to reconstruct the density
            matrix. The possible values for basis_change_string are
            given by the measurement basis used. For 1-qubit tomography
            experiment, state can be either "0" or "1".
        basis: the measurement basis used for tomography. Should be the
            Pauli basis.
    """
    assert (
        basis.name == "pauli"
    ), "Pauli reconstruction method is only implemented for Pauli measurements."

    density_matrices: list[numpy.ndarray] = []
    for freqs in frequencies:
        density_matrix: numpy.ndarray = (
            _constants.I
            + (2 * freqs["bcH"].get(0, 0) - 1) * _constants.X
            + (2 * freqs["bcSdgH"].get(0, 0) - 1) * _constants.Y
            + (2 * freqs["bcI"].get(0, 0) - 1) * _constants.Z
        ) / 2
        density_matrices.append(density_matrix)
    return density_matrices


def post_process_tomography_results_pauli(
    result: Result,
    tomographied_circuit: QuantumCircuit,
    basis: BaseMeasurementBasis,
    qubit_number: int = 1,
) -> list[numpy.ndarray]:
    """Compute and return the density matrix computed via state tomography.

    Args:
        result: the Result instance returned by the QPU after executing
            all the circuits returned by the
            one_qubit_tomography_circuits function.
        tomographied_circuit: the quantum circuit instance that is
            currently tomographied. Used to recover the circuit name.
        basis: the basis in which the measurements will be done.
        qubit_number: the number of qubits the parallel 1-qubit
            tomography should be performed on. Default to 1, i.e. no
            parallel execution.

    Returns:
        the 2 by 2 density matrix representing the prepared quantum
        state.
    """
    frequencies: list[dict[str, Counts]] = compute_frequencies(
        result, tomographied_circuit, basis, qubit_number
    )

    return frequencies_to_pauli_reconstruction(frequencies, basis)
