import abc
import typing as ty

import numpy
from qiskit import BasicAer, QuantumCircuit, execute

from qtom import _constants
from qtom.fit._helpers import _couter


class BaseMeasurementBasis(abc.ABC):
    """Base class for quantum tomography basis."""

    def __init__(self, name: str):
        """Initialise the basis with the given name."""
        self._name: str = name
        self._projector_states: ty.List[numpy.ndarray] = list()
        self._projectors: ty.List[ty.Tuple[numpy.ndarray, numpy.ndarray]] = list()

    @abc.abstractmethod
    def basis_change_circuits(self) -> ty.Iterable[QuantumCircuit]:
        """Return an interable on the basis change QuantumCircuit instances.

        A measurement basis is represented by the different basis change
        circuits that are needed to perform the different measurements in
        the desired basis.

        :return: an interable on the basis change QuantumCircuit instance.
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
    @abc.abstractmethod
    def qubit_number(self) -> int:
        """Return the number of qubits the basis is defined on."""
        pass

    @property
    def projector_states(self) -> ty.Iterable[numpy.ndarray]:
        """Return the states that represent the measurement basis.

        This method returns the states |phi_i> such that the POVM
        implemented by this basis is:

        U_{i in range(self.size)} {|phi_i><phi_i|, I - |phi_i><phi_i|}

        This only works for 1-qubit basis for the moment.
        """
        assert self.qubit_number == 1, "projector_states only works for 1-qubit basis."
        if not self._projector_states:
            simulator = BasicAer.get_backend("statevector_simulator")
            for basis_change_circuit in self.basis_change_circuits():
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

        This only works for 1-qubit basis for the moment.

        :return: the 1-qubit projectors implemented by this basis.
        """
        assert self.qubit_number == 1, "projectors only works for 1-qubit basis."
        if not self._projectors:
            for state in self.projector_states:
                self._projectors.append(
                    (_couter(state, state), _constants.I - self._projectors[-1])
                )

        yield from self._projectors
