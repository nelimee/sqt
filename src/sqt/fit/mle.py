import typing as ty

import numpy
from qiskit import BasicAer, QuantumCircuit, execute
from qiskit.result import Result
from scipy.optimize import Bounds, minimize

from sqt import _constants
from sqt.basis import BaseMeasurementBasis, PauliMeasurementBasis
from sqt.fit._helpers import (
    _bloch_vector_to_density_matrix,
    _couter,
    compute_frequencies,
)


def frequencies_to_mle_reconstruction(
    frequencies: ty.Dict[str, ty.Dict[str, float]],
    basis: BaseMeasurementBasis,
) -> numpy.ndarray:
    """Compute the density matrix from the given frenquencies.

    This function uses the Maximum Likelyhood Estimation method. It feeds the
    MLE cost function along with its gradient to scipy.optimize.minimize SLSQP
    non-linear optimiser.

    The MLE cost function has been modified to return a huge value when the
    trial state is non-physical (i.e. outside of the Bloch sphere).
    Moreover, the MLE cost function gradient has also been modified to point
    towards the center of the Bloch sphere if a non-physical state occurs.

    :param frenquencies: the estimated frequencies as a mapping
        {basis_change_str -> {state -> frequency}} where basis_change_str is
        the name of the quantum circuit performing the basis change, state is
        either "0" or "1" for 1-qubit and frequency is the estimated frequency.
    :param basis: the tomography basis used.
    :return: the reconstructed density matrix.
    """
    # Build the projectors and the observed frequencies
    simulator = BasicAer.get_backend("statevector_simulator")
    projectors: ty.List[numpy.ndarray] = list()
    observed_frequencies: ty.List[float] = list()
    for basis_change_circuit in basis.basis_change_circuits():
        inverted_basis_change_circuit = basis_change_circuit.inverse()
        state: numpy.ndarray = (
            execute(inverted_basis_change_circuit, simulator)
            .result()
            .get_statevector(inverted_basis_change_circuit.name)
        )
        projector = _couter(state, state)
        projectors.append(projector)
        projectors.append((_constants.I - projector))
        observed_frequencies.append(frequencies[basis_change_circuit.name].get("0", 0))
        observed_frequencies.append(frequencies[basis_change_circuit.name].get("1", 0))

    def inverse_likelyhood(s: numpy.ndarray) -> float:
        # Avoir non-physical states that confuse the optimiser
        if numpy.linalg.norm(s) > 1:
            return float("inf")
        accumulation = 0
        rho = _bloch_vector_to_density_matrix(s)
        for freq, proj in zip(observed_frequencies, projectors):
            accumulation += freq * numpy.log(numpy.trace(rho @ proj))
        return -numpy.real_if_close(accumulation)

    def inverse_likelyhood_grad(s: numpy.ndarray) -> numpy.ndarray:
        # If we have a non-physical state, get back within the Bloch sphere!
        if numpy.linalg.norm(s) > 1:
            return -s
        rho = _bloch_vector_to_density_matrix(s)
        grad_density = numpy.zeros_like(rho)
        for freq, proj in zip(observed_frequencies, projectors):
            grad_density += proj.T * freq / numpy.trace(rho @ proj)
        grad_density = -grad_density
        grad = numpy.real_if_close(
            numpy.array(
                [
                    numpy.trace(grad_density @ _constants.X),
                    numpy.trace(grad_density @ _constants.Y),
                    numpy.trace(grad_density @ _constants.Z),
                ]
            )
        )
        return grad

    ineq_constraint = {
        "type": "ineq",
        "fun": lambda s: numpy.array([1 - s[0] ** 2 - s[1] ** 2 - s[2] ** 2]),
        "jac": lambda s: numpy.array([[-2 * s[0], -2 * s[1], -2 * s[2]]]),
    }

    # Solving the system
    result = minimize(
        inverse_likelyhood,
        x0=numpy.zeros(3),
        jac=inverse_likelyhood_grad,
        constraints=[ineq_constraint],
        method="SLSQP",
        options={"ftol": 1e-9},
    )
    return _bloch_vector_to_density_matrix(result.x)


def post_process_tomography_results_mle(
    result: Result,
    tomographied_circuit: QuantumCircuit,
    qubit_index: ty.Optional[int] = None,
    is_parallel: bool = False,
    basis: ty.Optional[BaseMeasurementBasis] = None,
) -> numpy.ndarray:
    """
    Compute and return the density matrix computed via state tomography.

    This function uses the Maximum Likelyhood Estimation method.

    :param result: the Result instance returned by the QPU after executing all
        the circuits returned by the one_qubit_tomography_circuits function.
    :param tomographied_circuit: the quantum circuit instance that is currently
        tomographied. Used to recover the circuit name.
    :param qubit_index: index of the qubit used to perform the tomography
        experiments. If is_parallel is True, this is the index of the qubit
        that will be post-processed.
    :param is_parallel: True if the given Result instance has been obtained
        from a parallel execution, else False. If set to True, qubit_index
        should be set to the index of the qubit we want the results on.
    :return: the 2 by 2 density matrix representing the prepared quantum state.
    """
    if basis is None:
        basis = PauliMeasurementBasis()
    # Compute the frequencies
    frequencies: ty.Dict[str, ty.Dict[str, float]] = compute_frequencies(
        result, tomographied_circuit, qubit_index, is_parallel, basis
    )
    return frequencies_to_mle_reconstruction(frequencies, basis)
