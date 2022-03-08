import numpy
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator


def get_one_qubit_exact_density_matrix(
    circuit: QuantumCircuit,
) -> numpy.ndarray:
    """
    Compute and return the exact density matrix.

    :param circuit: the quantum circuit instance that is currently
        tomographied. Used to recover the circuit name.
    :return: the 2 by 2 density matrix representing the prepared quantum state.
    """
    simulator = AerSimulator(method="density_matrix")
    circuit_copy: QuantumCircuit = circuit.copy()
    circuit_copy.save_density_matrix()

    density_matrix_result = simulator.run(circuit_copy).result()
    return density_matrix_result.results[0].data.density_matrix
