import typing as ty

import numpy
from qiskit import QuantumCircuit

from .base import BaseMeasurementBasis


class TetrahedralMeasurementBasis(BaseMeasurementBasis):
    """1-qubit tetrahedral basis.

    The tetrahedral basis is the only SIC-POVM when only one qubit is
    considered. It is described in
    [this Wikipedia page](https://en.wikipedia.org/wiki/SIC-POVM#Simplest_example).
    """

    def __init__(self):
        """Initialise the tetrahedral basis."""
        super().__init__("tetrahedral")

    def basis_change_circuits(self) -> ty.List[QuantumCircuit]:
        """Return the 4 basis changes needed to perform tomography.

        Quantum state tomography is performed by measuring the quantum state
        in 4 different basis. This function returns 4 different quantum circuits
        that should be executed just before the Z-measurement to change the
        measurement basis.

        Naming convention: the returned quantum circuits are named as follow:
        - The name always starts with "bc" that stands for "basis change".
        -

        :return: all the basis change needed to perform the state tomography
            process.
        """
        basis_changes: ty.List[QuantumCircuit] = [
            QuantumCircuit(1, name="bcI"),
            QuantumCircuit(1, name="bcRyRz(0)"),
            QuantumCircuit(1, name="bcRyRz(2/3)"),
            QuantumCircuit(1, name="bcRyRz(4/3)"),
        ]
        basis_changes[0].id(0)
        theta: float = 2 * numpy.arccos(numpy.sqrt(1 / 3))
        for i in range(3):
            # Rz is OK up to a global phase, which is not important here.
            basis_changes[i + 1].u1(-2 * numpy.pi * i / 3, 0)
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
