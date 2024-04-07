import numpy
from qiskit import QuantumCircuit
from qiskit.result import Result

from sqt.basis.base import BaseMeasurementBasis
from sqt.circuits import get_parallelised_circuit_name, get_tomography_circuit_name
from sqt.counts import Counts

NumericType = int | float | numpy.number


def marginalise_all_counts(counts: Counts, qubit_number: int) -> list[Counts]:
    """Marginalise over all the qubits.

    Example:
        counts = Counts({
            0b001: 0.01,
            0b010: 0.09,
            0b110: 0.87,
            0b111: 0.03,
        })
        qubit_number = 3
        _marginalise_all_counts(counts, qubit_number) == [
            Counts({0: 0.96, 1: 0.04}),
            Counts({0: 0.01, 1: 0.99}),
            Counts({0: 0.1, 1: 0.9})
        ]

    Args:
        counts: the results obtained from the backend.
        qubit_number: the number of qubits measured. No key in counts
            should be strictly greater than ``2**qubit_number - 1``.

    Returns:
        a list of the marginalised counts for each qubits.
    """
    marginalised_counts: list[dict[int, float]] = [
        {0: 0, 1: 0} for _ in range(qubit_number)
    ]
    for measurement, probability in counts.items():
        for i in range(qubit_number):
            marginalised_counts[i][(measurement >> i) & 0b1] += probability
    return [Counts(c) for c in marginalised_counts]  # type: ignore


def compute_frequencies(
    result: Result,
    tomographied_circuit: QuantumCircuit,
    basis: BaseMeasurementBasis,
    qubit_number: int = 1,
) -> list[dict[str, Counts]]:
    """Compute an approximation of each expectation value with the frequency.

    This function takes as input the results of a given job, the circuit that
    have been tomographied (without the basis change nor the parallelisation)
    and returns the observed frequency for each measurement.

    Args:
        result: the data returned by IBM chips.
        tomographied_circuit: the quantum circuit we are interested in.
            Should be the QuantumCircuit instance **before** appending
            the tomography basis change. In other words, the name of
            this QuantumCircuit should not include the tomography basis
            identifier.
        basis: the tomography basis used.
        qubit_number: number of qubits the tomography process has been
            parallelised on. Default to 1, i.e. no parallelisation.

    Returns:
        the estimated frequencies as a list of mappings
        {basis_change_str -> {state -> frequency}} where
        basis_change_str is the name of the quantum circuit performing
        the basis change, state is either 0 or 1 and frequency is the
        estimated frequency.
    """
    # Compute the probabilities
    frequencies: list[dict[str, Counts]] = [dict() for _ in range(qubit_number)]
    tc_name: str = tomographied_circuit.name

    for basis_change_name in basis.basis_change_circuit_names:
        parallelised_circuit_name: str = get_parallelised_circuit_name(
            get_tomography_circuit_name(tc_name, basis_change_name),
            qubit_number,
        )
        counts: Counts = Counts(result.get_counts(parallelised_circuit_name))  # type: ignore
        counts_list: list[Counts] = marginalise_all_counts(counts, qubit_number)
        for qubit_index, qubit_counts in enumerate(counts_list):
            frequencies[qubit_index][basis_change_name] = qubit_counts
        # Change the counts in probabilities
        for qubit_index in range(qubit_number):
            count = sum(frequencies[qubit_index][basis_change_name].values())
            for state in frequencies[qubit_index][basis_change_name]:
                frequencies[qubit_index][basis_change_name][state] /= count
    return frequencies
