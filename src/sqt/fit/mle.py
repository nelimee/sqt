import numpy
from qiskit import QuantumCircuit
from qiskit.result import Result
from scipy.optimize import minimize

from sqt import _constants
from sqt._maths_helpers import bloch_vector_to_density_matrix
from sqt.basis.base import BaseMeasurementBasis
from sqt.counts import Counts
from sqt.fit._helpers import compute_frequencies


def frequencies_to_mle_reconstruction(
    frequencies: list[dict[str, Counts]],
    basis: BaseMeasurementBasis,
) -> list[numpy.ndarray]:
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
        observed_frequencies: list[float],
        projectors: list[numpy.ndarray],
    ) -> float:
        # Avoid non-physical states that confuse the optimiser
        penalty_factor: float = 0.0
        s_norm = float(numpy.linalg.norm(s))
        if s_norm > 1:
            # A simple
            #     return float('inf')
            # does not work here, probably because such a brutal and
            # non-continuous change confuses the optimiser. Instead,
            # we apply a very strong penalty for exerything that is
            # outside the Bloch sphere.
            penalty_factor = 1 - s_norm
            s /= s_norm
        accumulation = 0
        rho = bloch_vector_to_density_matrix(s)
        for freq, proj in zip(observed_frequencies, projectors):
            accumulation += freq * numpy.log(numpy.trace(rho @ proj))
        return -numpy.real_if_close(accumulation) + (
            1 - numpy.exp(1e10 * penalty_factor)
        )

    def inverse_likelyhood_grad(
        s: numpy.ndarray,
        observed_frequencies: list[float],
        projectors: list[numpy.ndarray],
    ) -> numpy.ndarray:
        # If we have a non-physical state, get back within the Bloch sphere!
        if numpy.linalg.norm(s) > 1:
            return -s
        rho = bloch_vector_to_density_matrix(s)
        grad_density = numpy.zeros_like(rho)
        for freq, proj in zip(observed_frequencies, projectors):
            grad_density -= proj * freq / numpy.trace(rho @ proj)
        # The gradient projection is non-trivial and involves the adjoint of
        # the basis change Jacobian. The formula below has been checked and
        # should be correct for any gradient.
        grad = numpy.real_if_close(
            numpy.array(
                [
                    numpy.trace(grad_density @ _constants.X),
                    numpy.trace(grad_density @ _constants.Y),
                    numpy.trace(grad_density @ _constants.Z),
                ]
            )
            / 2
        )
        return grad

    ineq_constraint = {
        "type": "ineq",
        "fun": lambda s: numpy.array([1 - s[0] ** 2 - s[1] ** 2 - s[2] ** 2]),
        "jac": lambda s: numpy.array([[-2 * s[0], -2 * s[1], -2 * s[2]]]),
    }

    density_matrices: list[numpy.ndarray] = []
    # This reconstruction could potentially be performed in parallel.
    # Left as a TODO for the moment.
    for freqs in frequencies:
        # Build the projectors and the observed frequencies
        projectors: list[numpy.ndarray] = list()
        observed_frequencies: list[float] = list()
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
            options={"ftol": 1e-12},
        )
        density_matrix: numpy.ndarray = bloch_vector_to_density_matrix(result.x)
        density_matrices.append(density_matrix)
    return density_matrices


def post_process_tomography_results_mle(
    result: Result,
    tomographied_circuit: QuantumCircuit,
    basis: BaseMeasurementBasis,
    qubit_number: int = 1,
) -> list[numpy.ndarray]:
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
    frequencies: list[dict[str, Counts]] = compute_frequencies(
        result, tomographied_circuit, basis, qubit_number
    )
    return frequencies_to_mle_reconstruction(frequencies, basis)
