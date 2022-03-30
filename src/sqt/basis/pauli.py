import typing as ty

from qiskit import QuantumCircuit

from sqt.basis.base import BaseMeasurementBasis


class PauliMeasurementBasis(BaseMeasurementBasis):
    """Pauli basis used to perform quantum tomography."""

    def __init__(self):
        """Initialise the Pauli basis on 1 qubit."""
        super().__init__("pauli")

    @property
    def basis_change_circuits(self) -> ty.List[QuantumCircuit]:
        """Return the 3 basis change needed to perform tomography.

        Quantum state tomography is performed by measuring the quantum state
        in 3 different basis. This function returns 3 different quantum
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
        basis_changes: ty.List[QuantumCircuit] = [
            QuantumCircuit(1, name="bcI"),
            QuantumCircuit(1, name="bcH"),
            QuantumCircuit(1, name="bcSdgH"),
        ]
        # We should let this identity gate here to be sure that it will be
        # compiled used the standard Rz sx Rz sx Rz decomposition.
        basis_changes[0].id(0)
        basis_changes[1].h(0)
        basis_changes[2].sdg(0)
        basis_changes[2].h(0)

        return basis_changes

    @property
    def size(self) -> int:
        """Return the number of basis change circuits in the basis."""
        return 3
