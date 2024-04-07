import abc
import typing as ty

from qiskit import QuantumCircuit
from qiskit.quantum_info.states import Statevector
from qiskit_aer import AerSimulator

from sqt import _constants
from sqt._maths_helpers import couter
from sqt._typing import ComplexMatrix, QuantumState


class BaseMeasurementBasis(abc.ABC):
    """Base class for quantum tomography basis."""

    def __init__(self, name: str):
        """Initialise the basis with the given name."""
        self._name: str = name
        self._projector_states: list[QuantumState] = list()
        self._projectors: list[tuple[ComplexMatrix, ComplexMatrix]] = list()

    @property
    @abc.abstractmethod
    def basis_change_circuits(self) -> ty.Iterable[QuantumCircuit]:
        """Return an interable on the basis change QuantumCircuit instances.

        A measurement basis is represented by the different basis change
        circuits that are needed to perform the different measurements in
        the desired basis.

        Returns:
            an iterable on the basis change QuantumCircuit instances.
        """
        pass

    @property
    def name(self) -> str:
        """Name of the basis."""
        return self._name

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """Return the number of basis change circuits in the basis."""
        pass

    @property
    def projector_states(self) -> ty.Iterator[QuantumState]:
        """Return the states that represent the measurement basis.

        This method returns the states |phi_i> such that the POVM
        implemented by this basis is:

        U_{i in range(self.size)} {|phi_i><phi_i|, I - |phi_i><phi_i|}

        This only works for 1-qubit basis.

        Returns:
            the states that represent the measurement basis.
        """
        if not self._projector_states:
            simulator = AerSimulator()
            for basis_change_circuit in self.basis_change_circuits:
                inverted_basis_change_circuit = basis_change_circuit.inverse()
                #
                inverted_basis_change_circuit.save_statevector()  # type: ignore
                state = (
                    simulator.run(inverted_basis_change_circuit)
                    .result()
                    .get_statevector(inverted_basis_change_circuit.name)
                )
                self._projector_states.append(
                    state.astype(complex)
                    if not isinstance(state, Statevector)
                    else state.data
                )
        yield from self._projector_states

    @property
    def projectors(self) -> ty.Iterator[tuple[ComplexMatrix, ComplexMatrix]]:
        """Return the POVM projectors used for this basis.

        Each tuple corresponds to one projection basis, with the first entry of
        the tuple being a projector on a given state and the second entry being
        an orthogonal projector.

        This only works for 1-qubit basis.

        Returns:
            the 1-qubit projectors implemented by this basis.
        """
        if not self._projectors:
            for state in self.projector_states:
                proj: ComplexMatrix = couter(state, state)
                orth: ComplexMatrix = _constants.I - proj
                self._projectors.append((proj, orth))
        yield from self._projectors

    @property
    def basis_change_circuit_names(self) -> ty.Iterator[str]:
        yield from (c.name for c in self.basis_change_circuits)
