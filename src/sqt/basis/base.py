import abc
import typing as ty

import numpy
from qiskit import BasicAer, QuantumCircuit, execute

from sqt import _constants
from sqt._maths_helpers import couter


class BaseMeasurementBasis(abc.ABC):
    """Base class for quantum tomography basis."""

    def __init__(self, name: str):
        """Initialise the basis with the given name."""
        self._name: str = name
        self._projector_states: ty.List[numpy.ndarray] = list()
        self._projectors: ty.List[ty.Tuple[numpy.ndarray, numpy.ndarray]] = list()

    @property
    @abc.abstractmethod
    def basis_change_circuits(self) -> ty.Iterable[QuantumCircuit]:
        """Return an interable on the basis change QuantumCircuit instances.

        A measurement basis is represented by the different basis change
        circuits that are needed to perform the different measurements in
        the desired basis.

        :return: an iterable on the basis change QuantumCircuit instances.
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
    def projector_states(self) -> ty.Iterable[numpy.ndarray]:
        """Return the states that represent the measurement basis.

        This method returns the states |phi_i> such that the POVM
        implemented by this basis is:

        U_{i in range(self.size)} {|phi_i><phi_i|, I - |phi_i><phi_i|}

        This only works for 1-qubit basis.
        """
        if not self._projector_states:
            simulator = BasicAer.get_backend("statevector_simulator")
            for basis_change_circuit in self.basis_change_circuits:
                inverted_basis_change_circuit = basis_change_circuit.inverse()
                state: numpy.ndarray = (
                    execute(inverted_basis_change_circuit, simulator)
                    .result()
                    .get_statevector(inverted_basis_change_circuit.name)
                )
                self._projector_states.append(state)
        yield from self._projector_states

    @property
    def projectors(self) -> ty.Iterable[ty.Tuple[numpy.ndarray, numpy.ndarray]]:
        """Return the POVM projectors used for this basis.

        Each tuple corresponds to one projection basis, with the first entry of
        the tuple being a projector on a given state and the second entry being
        an orthogonal projector.

        This only works for 1-qubit basis.

        :return: the 1-qubit projectors implemented by this basis.
        """
        if not self._projectors:
            for state in self.projector_states:
                proj: numpy.ndarray = couter(state, state)
                orth: numpy.ndarray = _constants.I - proj
                self._projectors.append((proj, orth))
        yield from self._projectors

    @property
    def basis_change_circuit_names(self) -> ty.Iterator[str]:
        yield from (c.name for c in self.basis_change_circuits)
