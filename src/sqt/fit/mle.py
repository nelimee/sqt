import typing as ty

import numpy
from qiskit import QuantumCircuit
from qiskit.result import Result
from scipy.optimize import minimize

from sqt import _constants
from sqt.basis.base import BaseMeasurementBasis
from sqt.fit._helpers import compute_frequencies
from sqt._maths_helpers import bloch_vector_to_density_matrix
from sqt.counts import Counts


def frequencies_to_mle_reconstruction(
    frequencies: ty.List[ty.Dict[str, Counts]],
    basis: BaseMeasurementBasis,
) -> ty.List[numpy.ndarray]:
    """Compute the density matrix from the given frenquencies.

    This function uses the Maximum Likelyhood Estimation method. It feeds the
    MLE cost function along with its gradient to scipy.optimize.minimize SLSQP
    non-linear optimiser.

    The MLE cost function has been modified to return a huge value when the
    trial state is non-physical (i.e. outside of the Bloch sphere).
    Moreover, the MLE cost function gradient has also been modified to point
    towards the center of the Bloch sphere if a non-physical state occurs.

    :param frenquencies: the estimated frequencies as a list of mappings
        {basis_change_str -> {state -> frequency}} where basis_change_str is
        the name of the quantum circuit performing the basis change, state is
        either "0" or "1" for 1-qubit and frequency is the estimated frequency.
    :param basis: the tomography basis used.
    :return: the reconstructed density matrix.
    """

    def inverse_likelyhood(
        s: numpy.ndarray,
        observed_frequencies: ty.List[float],
        projectors: ty.List[numpy.ndarray],
    ) -> float:
        # Avoid non-physical states that confuse the optimiser
        if numpy.linalg.norm(s) > 1:
            # TODO: make this a little bit smoother.
            # Idea: project s on the sphere, do the computation, and add a penalty
            # that scales with the distance outside of the sphere.
            return float("inf")
        accumulation = 0
        rho = bloch_vector_to_density_matrix(s)
        for freq, proj in zip(observed_frequencies, projectors):
            accumulation += freq * numpy.log(numpy.trace(rho @ proj))
        return -numpy.real_if_close(accumulation)

    def inverse_likelyhood_grad(
        s: numpy.ndarray,
        observed_frequencies: ty.List[float],
        projectors: ty.List[numpy.ndarray],
    ) -> numpy.ndarray:
        # If we have a non-physical state, get back within the Bloch sphere!
        if numpy.linalg.norm(s) > 1:
            return -s
        rho = bloch_vector_to_density_matrix(s)
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

    density_matrices: ty.List[numpy.ndarray] = []
    # This reconstruction could potentially be performed in parallel.
    # Left as a TODO for the moment.
    for freqs in frequencies:
        # Build the projectors and the observed frequencies
        projectors: ty.List[numpy.ndarray] = list()
        observed_frequencies: ty.List[float] = list()
        for basis_change_name, (state_projector, orthogonal_projector) in zip(
            basis.basis_change_circuit_names, basis.projectors
        ):
            projectors.append(state_projector)
            observed_frequencies.append(freqs[basis_change_name].get(0, 0))
            projectors.append(orthogonal_projector)
            observed_frequencies.append(freqs[basis_change_name].get(1, 0))
        # Solving the system
        result = minimize(
            lambda s: inverse_likelyhood(s, observed_frequencies, projectors),
            x0=numpy.zeros(3),
            jac=lambda s: inverse_likelyhood_grad(s, observed_frequencies, projectors),
            constraints=[ineq_constraint],
            method="SLSQP",
            options={"ftol": 1e-9},
        )
        print(result.__dict__)
        density_matrix: numpy.ndarray = bloch_vector_to_density_matrix(result.x)
        density_matrices.append(density_matrix)
    return density_matrices


def post_process_tomography_results_mle(
    result: Result,
    tomographied_circuit: QuantumCircuit,
    basis: BaseMeasurementBasis,
    qubit_number: int = 1,
) -> ty.List[numpy.ndarray]:
    """
    Compute and return the density matrix computed via state tomography.

    This function uses the Maximum Likelyhood Estimation method.

    :param result: the Result instance returned by the QPU after executing all
        the circuits returned by the one_qubit_tomography_circuits function.
    :param tomographied_circuit: the quantum circuit instance that is currently
        tomographied. Used to recover the circuit name.
    :param basis: the basis in which the measurements will be done.
    :param qubit_number: the number of qubits the parallel 1-qubit tomography
        should be performed on. Default to 1, i.e. no parallel execution.
    :return: the 2 by 2 density matrix representing the prepared quantum state.
    """
    # Compute the frequencies
    frequencies: ty.List[ty.Dict[str, Counts]] = compute_frequencies(
        result, tomographied_circuit, basis, qubit_number
    )
    return frequencies_to_mle_reconstruction(frequencies, basis)
