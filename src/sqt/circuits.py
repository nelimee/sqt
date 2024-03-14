from qiskit import QuantumCircuit

from sqt.basis.base import BaseMeasurementBasis

TOMOGRAPHY_CIRCUIT_NAME_FORMAT: str = (
    "{tomographied_circuit_name}_{basis_change_circuit_name}"
)
PARALLELISED_CIRCUIT_NAME_FORMAT: str = "{base_circuit_name}_{qubit_number}"


def get_tomography_circuit_name(name: str, basis_change_name: str) -> str:
    return TOMOGRAPHY_CIRCUIT_NAME_FORMAT.format(
        tomographied_circuit_name=name, basis_change_circuit_name=basis_change_name
    )


def get_parallelised_circuit_name(base_name: str, qubit_number: int) -> str:
    return PARALLELISED_CIRCUIT_NAME_FORMAT.format(
        base_circuit_name=base_name, qubit_number=qubit_number
    )


def _parallelise_one_qubit_tomography_circuits(
    one_qubit_quantum_circuits: list[QuantumCircuit],
    qubit_number: int,
) -> list[QuantumCircuit]:
    """Return the quantum circuits needed to perform a state tomography.

    This function duplicates the circuits given in one_qubit_quantum_circuits
    to execute them in parallel on qubit_number qubits.

    :param one_qubit_quantum_circuits: a list of quantum circuits that will be
        parallelised over qubit_number qubits.
    :param qubit_number: the number of qubits the parallel 1-qubit tomography
        should be performed on.
    :return: the quantum circuits that should be executed to perform the state
        tomography.
    """
    quantum_circuits: list[QuantumCircuit] = list()
    for circuit in one_qubit_quantum_circuits:
        qc = QuantumCircuit(
            qubit_number,
            qubit_number,
            name=get_parallelised_circuit_name(circuit.name, qubit_number),
        )
        for qubit_index in range(qubit_number):
            qc.compose(
                circuit, inplace=True, qubits=[qubit_index], clbits=[qubit_index]
            )
        quantum_circuits.append(qc)
    return quantum_circuits


def one_qubit_tomography_circuits(
    tomographied_circuit: QuantumCircuit,
    basis: BaseMeasurementBasis,
    qubit_number: int = 1,
    add_barrier: bool = True,
) -> list[QuantumCircuit]:
    """Return the quantum circuits needed to perform a state tomography.

    This function appends each of the basis change given by the basis to the
    given quantum circuit to produce the quantum circuits that should be
    executed to perform the 1-qubit quantum state tomography.

    :param tomographied_circuit: a quantum circuit that prepares the state to
        be tomographied.
    :param qubit_number: the number of qubits the parallel 1-qubit tomography
        should be performed on. Default to 1, i.e. no parallel execution.
    :param basis: the basis in which the measurements will be done.
    :param add_barrier: if True, add a barrier between the state preparation
        and the measurement basis change.
    :return: the quantum circuits that should be executed to perform the state
        tomography in the given basis.
    """
    quantum_circuits: list[QuantumCircuit] = list()

    for basis_change_circuit in basis.basis_change_circuits:
        qc = QuantumCircuit(
            1,
            1,
            name=get_tomography_circuit_name(
                tomographied_circuit.name, basis_change_circuit.name
            ),
        )
        qc.compose(tomographied_circuit, inplace=True)
        if add_barrier:
            qc.barrier()
        qc.compose(basis_change_circuit, inplace=True)
        qc.measure(0, 0)
        quantum_circuits.append(qc)
    quantum_circuits = _parallelise_one_qubit_tomography_circuits(
        quantum_circuits, qubit_number
    )

    return quantum_circuits
