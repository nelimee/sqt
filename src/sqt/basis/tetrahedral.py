import numpy
from qiskit import QuantumCircuit

from sqt.basis.base import BaseMeasurementBasis


class TetrahedralMeasurementBasis(BaseMeasurementBasis):
    """1-qubit tetrahedral basis.

    The tetrahedral basis is the only SIC-POVM when only one qubit is
    considered. It is described in
    [this Wikipedia page](https://en.wikipedia.org/wiki/SIC-POVM#Simplest_example).
    """

    def __init__(self):
        """Initialise the tetrahedral basis."""
        super().__init__("tetrahedral")

    @property
    def basis_change_circuits(self) -> list[QuantumCircuit]:
        """Return the 4 basis changes needed to perform tomography.

        Quantum state tomography is performed by measuring the quantum state
        in 4 different basis. This function returns 4 different quantum circuits
        that should be executed just before the Z-measurement to change the
        measurement basis.

        Naming convention: the returned quantum circuits are named as follow:
        - The name always starts with "bc" that stands for "basis change".
        - Then follows a string representation of the gates used to change the
          measurement basis. If a H gate is applied, the circuit name is "bcH".
          If no gate is applied, the circuit name is "bcI". When several gates
          are applied, they are order in the left-to-right order in the quantum
          circuit notation, i.e. the first gate applied comes first in the name.
        - "dg" stands for "dagger", i.e. the complex conjuguate.

        Returns:
            all the basis change needed to perform the state tomography
            process.
        """
        basis_changes: list[QuantumCircuit] = [
            QuantumCircuit(1, name="bcI"),
            QuantumCircuit(1, name="bcRyRz(0)"),
            QuantumCircuit(1, name="bcRyRz(2/3)"),
            QuantumCircuit(1, name="bcRyRz(4/3)"),
        ]
        # We should let this identity gate here to be sure that it will be
        # compiled used the standard Rz sx Rz sx Rz decomposition.
        basis_changes[0].id(0)
        theta: float = 2 * numpy.arccos(numpy.sqrt(1 / 3))
        for i in range(3):
            basis_changes[i + 1].p(-2 * numpy.pi * i / 3, 0)
            basis_changes[i + 1].ry(-theta, 0)
        return basis_changes

    @property
    def size(self) -> int:
        """Return the number of basis change circuits in the basis."""
        return 4

    @property
    def qubit_number(self) -> int:
        """Return the number of qubits the basis is defined on."""
        return 1
