import typing as ty

from qiskit import QuantumCircuit

from .base import BaseMeasurementBasis


class PauliMeasurementBasis(BaseMeasurementBasis):
    """Pauli basis used to perform quantum tomography."""

    def __init__(self, qubit_number: int = 1):
        """Initialise the Pauli basis on qubit_number qubits."""
        super().__init__("pauli")
        self._qubit_number: int = qubit_number

    def basis_change_circuits(self) -> ty.List[QuantumCircuit]:
        """Return the 4^n basis change needed to perform tomography.

        Quantum state tomography is performed by measuring the quantum state
        in 4^n different basis. This function returns 4^n different quantum
        circuits that should be executed just before the Z-measurements to
        change the measurement basis.

        Naming convention: the returned quantum circuits are named as follow:
        - The name always starts with "bc" that stands for "basis change".
        - Then follows a string representation of the gates used to change the
          measurement basis. If a H gate is applied, the circuit name is "bcH".
          If no gate is applied, the circuit name is "bcI". When several gates
          are applied, they are order in the left-to-right order in the quantum
          circuit notation, i.e. the first gate applied comes first in the name.
        - "dg" stands for "dagger", i.e. the complex conjuguate.

        :return: all the basis change needed to perform the state tomography
            process.
        """
        assert (
            self.qubit_number > 1
        ), "Pauli basis is only implemented for 1 qubit for the moment."
        basis_changes: ty.List[QuantumCircuit] = [
            QuantumCircuit(1, name="bcI"),
            QuantumCircuit(1, name="bcH"),
            QuantumCircuit(1, name="bcSdgH"),
        ]

        basis_changes[1].h(0)
        basis_changes[2].sdg(0)
        basis_changes[2].h(0)

        return basis_changes

    @property
    def size(self) -> int:
        """Return the number of basis change circuits in the basis."""
        return 4 ** self.qubit_number - 1

    @property
    def qubit_number(self) -> int:
        """Return the number of qubits the basis is defined on."""
        return self._qubit_number
