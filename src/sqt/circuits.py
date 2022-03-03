import typing as ty

from qiskit import QuantumCircuit

from sqt.basis import BaseMeasurementBasis


def _parallelise_one_qubit_tomography_circuits(
    one_qubit_quantum_circuits: ty.List[QuantumCircuit],
    qubit_number: int,
) -> ty.List[QuantumCircuit]:
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
    quantum_circuits: ty.List[QuantumCircuit] = list()
    for circuit in one_qubit_quantum_circuits:
        qc = QuantumCircuit(qubit_number, qubit_number, name=circuit.name)
        for qubit_index in range(qubit_number):
            qc.compose(
                circuit, inplace=True, qubits=[qubit_index], clbits=[qubit_index]
            )
        quantum_circuits.append(qc)
    return quantum_circuits


def one_qubit_tomography_circuits(
    tomographied_circuit: QuantumCircuit,
    basis: BaseMeasurementBasis,
    is_parallel: bool = False,
    qubit_number: ty.Optional[int] = None,
) -> ty.List[QuantumCircuit]:
    """Return the quantum circuits needed to perform a state tomography.

    This function appends each of the basis change given by the basis to the
    given quantum circuit to produce the quantum circuits that should be
    executed to perform the 1-qubit quantum state tomography.

    :param tomographied_circuit: a quantum circuit that prepares the state to
        be tomographied.
    :param is_parallel: if True, the 1-qubit tomography experiment is
        duplicated in parallel on qubit_number qubits. Else, it is only
        performed on 1 qubit.
    :param qubit_number: the number of qubits the parallel 1-qubit tomography
        should be performed on. Not used if is_parallel is False. Should be
        different from None if is_parallel is True.
    :param basis: the basis in which the measurements will be done.
    :return: the quantum circuits that should be executed to perform the state
        tomography in the given basis.
    """
    quantum_circuits: ty.List[QuantumCircuit] = list()

    for basis_change_circuit in basis.basis_change_circuits():
        qc = QuantumCircuit(
            1, 1, name=f"{tomographied_circuit.name}_{basis_change_circuit.name}"
        )
        qc.compose(tomographied_circuit, inplace=True)
        qc.compose(basis_change_circuit, inplace=True)
        qc.measure(0, 0)
        quantum_circuits.append(qc)
    if is_parallel:
        assert (
            qubit_number is not None
        ), "qubit_number should not be None when is_parallel is True."
        quantum_circuits = _parallelise_one_qubit_tomography_circuits(
            quantum_circuits, qubit_number
        )

    return quantum_circuits
