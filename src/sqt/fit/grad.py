"""Implement a projected gradient method from https://arxiv.org/pdf/1609.07881.pdf.

This module implements a projected gradient descent to minimise the Maximum Likelihood
Estimator (MLE).

The projection is done according to Appendix C.1. of the main paper linked above
and has been inspired by the code
https://github.com/lanl/quantum_algorithms/blob/master/subroutines/quantum_tomography/qtml.jl

The gradient descent currently does not implement line-search due to issues described
in the line_search function. Using the algorithm described in 
https://sites.math.washington.edu/~burke/crs/408/notes/nlp/gpa.pdf
might solve the issue, in which case the convergence speed will likely be much better.
"""

import typing as ty

import numpy
from qiskit import QuantumCircuit
from qiskit.result import Result

from sqt.basis.base import BaseMeasurementBasis
from sqt.fit._helpers import compute_frequencies
from sqt._maths_helpers import couter, get_orthogonal_state
from sqt.counts import Counts


def frobenius_inner_product(A: numpy.ndarray, B: numpy.ndarray) -> numpy.ndarray:
    """Return the Frobenius inner product between the two given arrays.

    :param A: a 2-dimensional matrix.
    :param B: a 2-dimensional matrix.
    :return: the Frobenius inner product between A and B.
    """
    return numpy.inner(A.T.conj().ravel(), B.ravel())


def negative_log_likelyhood(
    density_matrix: numpy.ndarray,
    projector_matrices: ty.List[numpy.ndarray],
    observed_frequencies: numpy.ndarray,
) -> float:
    """Compute the negative log likelyhood value for the given parameters.

    :param density_matrix: the trial density matrix whose quality should be
        estimated using the negative log likelyhood formula.
    :param projector_matrices: the projectors used to measure the given
        observed frequencies.
    :param observed_frequencies: the frequencies computed from the hardware
        returns.
    :return: the negative log likelyhood of the given density matrix.
    """
    result = -sum(
        observed_frequencies[k]
        * numpy.log(numpy.trace(density_matrix @ projector_matrices[k]))
        for k in range(len(observed_frequencies))
    )
    return result


def negative_log_likelyhood_gradient(
    density_matrix: numpy.ndarray,
    projector_vectors: ty.List[numpy.ndarray],
    projector_matrices: ty.List[numpy.ndarray],
    observed_frequencies: numpy.ndarray,
):
    """Compute the negative log likelyhood gradient for the given parameters.

    :param density_matrix: the trial density matrix whose quality should be
        estimated using the negative log likelyhood formula.
    :param projector_vectors: the states we project in before each measurement.
        Each entry in projector_matrices can be computed from projector_vectors
        but we require both as parameters to save computations.
    :param projector_matrices: the projectors used to measure the given
        observed frequencies.
    :param observed_frequencies: the frequencies computed from the hardware
        returns.
    :return: the negative log likelyhood of the given density matrix.
    """
    projector_number: int = len(projector_vectors)
    gradient = numpy.zeros_like(density_matrix)
    for k in range(projector_number):
        freq = observed_frequencies[k]
        proj_vec = projector_vectors[k]
        denominator = numpy.vdot(proj_vec, density_matrix @ proj_vec)
        if numpy.real(denominator) >= 1e-6:
            gradient -= (freq / denominator) * projector_matrices[k].T
    return gradient


def project(matrix: numpy.ndarray) -> numpy.ndarray:
    """Project the given hermitian matrix to the density matrix space."""
    # Eigenvalues are real numbers, sorted in ascending order
    eigvals, eigvecs = numpy.linalg.eigh(matrix)
    # Make everything in descending order
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    mu = eigvals[0]
    dimension = numpy.size(eigvals)
    # Eigenvalue counter needed in case the following loop is executed
    # until the very end, in this case eigenvalue_counter == max_eig_index+1
    eigenvalue_counter = 1
    for max_eig_index in range(1, dimension):
        lambd = eigvals[max_eig_index]
        if (max_eig_index + 1) * lambd >= mu + lambd - 1:
            mu += lambd
            eigenvalue_counter += 1
        else:
            break
    mu = (mu - 1) / eigenvalue_counter
    # Re-construct the density matrix with only the most significant
    # eigenvalues.
    density_matrix = numpy.zeros((dimension, dimension), dtype=numpy.complex_)
    for j in range(eigenvalue_counter):
        weight = eigvals[j] - mu
        eigvector = eigvecs[:, j] / numpy.linalg.norm(eigvecs[:, j])
        density_matrix += weight * numpy.outer(eigvector.T.conj(), eigvector).T
    return density_matrix


def line_search(
    density_matrix: numpy.ndarray,
    projector_matrices: ty.List[numpy.ndarray],
    observed_frequencies: numpy.ndarray,
    gradient: numpy.ndarray,
    previous_gradient_step: float,
    alpha: float,
    beta: float,
) -> float:
    """Perform a backtracking line search to find the best gradient step.

    The backtracking line search try to find a good gradient step (or learning
    rate) in order to fullfil the Wolfe condition.

    :warning:
    This method is currently not working correctly and systematically returns
    a value of 0.001 that has been empirically determined as sufficiently small
    for the optimisation problems I had. This value might be too large for
    other optimisation problems.

    Some observations on this method: it seems like the line search does not
    work because there are cases where the projection makes the actual descent
    direction in the oposite direction of the gradient. In other words, if d
    is the descent direction, obtained with the formula

    d = proj(rho - gradient_step * gradient) - rho

    then there are cases where <d , gradient> < 0, i.e. d does not follow the
    gradient anymore but goes **backward**. I suspect this is due to some
    conditions that are not fulfilled by the problem (convexity, continuity,
    condition on the projection, ...?), but I do not have any proof yet.

    See https://sites.math.washington.edu/~burke/crs/408/notes/nlp/gpa.pdf

    :param density_matrix: the current trial density matrix.
    :param projector_matrices: the projectors used to measure the given
        observed frequencies.
    :param observed_frequencies: the frequencies computed from the hardware
        returns.
    :param gradient: the value of the log likelyhood gradient at the given
        density matrix.
    :param previous_gradient_step: the gradient step that has been used
        previously. Used as a starting point for the gradient step line search.
    :param alpha: unused parameter.
    :param beta: unused parameter.
    """
    return 0.001
    gradient_step: float = previous_gradient_step
    current_negative_log_likelyhood = negative_log_likelyhood(
        project(density_matrix), projector_matrices, observed_frequencies
    )

    new_density_matrix = project(density_matrix - gradient_step * gradient)
    descent_direction = new_density_matrix - density_matrix
    new_negative_log_likelyhood = negative_log_likelyhood(
        new_density_matrix, projector_matrices, observed_frequencies
    )
    while (
        new_negative_log_likelyhood
        > current_negative_log_likelyhood
        # - gradient_step * frobenius_inner_product(gradient, descent_direction)
        - gradient_step * numpy.linalg.norm(descent_direction, ord="fro") ** 2 / 2
    ):
        gradient_step *= beta
        new_density_matrix = project(density_matrix - gradient_step * gradient)
        descent_direction = new_density_matrix - density_matrix
        if frobenius_inner_product(gradient, descent_direction) >= 0:
            print("Continuing to lower down gradient_step")
            continue
        new_negative_log_likelyhood = negative_log_likelyhood(
            new_density_matrix, projector_matrices, observed_frequencies
        )
        print(f"Step = {gradient_step}")
    return gradient_step


def reconstruct_density_matrix(
    empirical_frequencies: ty.Union[numpy.ndarray, ty.List[float]],
    projector_vectors: ty.List[numpy.ndarray],
    max_iter: int = 10000,
    eps: float = 1e-6,
    alpha: float = 1e-4,
    beta: float = 0.5,
    verbose: bool = False,
    warning_threshold: float = 1e-3,
) -> numpy.ndarray:
    """
    Reconstruct the density matrix from observations using projected gradient descent.

    :param empirical_frequencies: the estimated frequencies as a mapping
        {basis_change_str -> {state -> frequency}} where basis_change_str is
        the name of the quantum circuit performing the basis change, state is
        either "0" or "1" for 1-qubit and frequency is the estimated frequency.
    :param projector_vectors: the states we project in before each measurement.
        Each entry in projector_matrices can be computed from projector_vectors
        but we require both as parameters to save computations.
    :max_iter: maximum number of iterations performed.
    :param eps: a threshold to stop the optimisation. If the previous descent
        step has a Frobenius norm lower than this threshold, the gradient
        descent is considered as converged and the result is returned without
        performing more descent iteration.
    :param alpha: parameter for the backtracking line search. See the
        line_search function documentation for more explanations.
    :param beta: parameter for the backtracking line search. See the
        line_search function documentation for more explanations.
    :param verbose: if True, messages about the current iteration and the
        convergence of the projected gradient descent are printed to stdout.
    :param warning_threshold: a threshold to warn about non-convergence.
        If the last descent step has a Frobenius norm higher than this
        threshold, the gradient descent is considered as non-converged and a
        warning is issued.
    :return: the density matrix that maximise the likelyhood cost function (or
        equivalently minimise the negative log likelyhood cost function).
    """
    # Make everything a numpy array.
    empirical_frequencies = numpy.asarray(empirical_frequencies)
    # Get some useful numbers that will be re-used accross the optimisation.
    dimension = numpy.size(projector_vectors[0])
    # Update the projectors to have normalised vectors
    projectors = [proj / numpy.linalg.norm(proj) for proj in projector_vectors]
    projector_matrices = [couter(proj, proj) for proj in projectors]
    gradient_step: float = 1.0
    movement_norm: float = 0.0
    density_matrix = numpy.eye(dimension, dtype=numpy.complex_) / dimension

    for it in range(max_iter):
        # Estimate the gradient for this iteration
        gradient = negative_log_likelyhood_gradient(
            density_matrix, projectors, projector_matrices, empirical_frequencies
        )
        # Line search for gradient descent:
        gradient_step = line_search(
            density_matrix,
            projector_matrices,
            empirical_frequencies,
            gradient,
            1,
            alpha,
            beta,
        )
        # Perform gradient descent
        updated_density_matrix = density_matrix - gradient_step * gradient

        # Normalise the matrix to avoid descent issues by removing the least
        # important eigenvalues in terms of magnitude and rescaling the other
        # eigenvalues.
        updated_density_matrix = project(updated_density_matrix)
        # Compute the difference between the previous density matrix estimate and
        # the new one. This is usefull for the stopping criterion.
        movement_norm = (
            numpy.linalg.norm(updated_density_matrix - density_matrix, ord="fro") / 2
        )
        density_matrix = updated_density_matrix

        if verbose:
            print(f"{it+1} / {max_iter}  --->  {movement_norm}", end="\r", flush=True)

        if movement_norm < eps:
            break
    if movement_norm > warning_threshold:
        print(f"Warning! Gradient descent finished with a step of {movement_norm}.")
    if verbose:
        cost = negative_log_likelyhood(
            density_matrix, projector_matrices, empirical_frequencies
        )
        print(
            f"Ended at ||rho' - rho|| = {movement_norm} in {it+1} iterations "
            f"with a cost of {cost}."
        )
    return density_matrix


def frequencies_to_grad_reconstruction(
    frequencies: ty.List[ty.Dict[str, Counts]],
    basis: BaseMeasurementBasis,
) -> ty.List[numpy.ndarray]:
    """Compute the density matrix from the given frenquencies.

    This function uses the Maximum Likelyhood Estimation method and a
    projected gradient descent to minimise the negative log likelyhood
    cost function. Details about the actual optimisation process can be
    found in the reconstruct_density_matrix function.

    :param frenquencies: the estimated frequencies as a list of mappings
        {basis_change_str -> {state -> frequency}} where basis_change_str is
        the name of the quantum circuit performing the basis change, state is
        either "0" or "1" for 1-qubit and frequency is the estimated frequency.
    :param basis: the tomography basis used.
    :return: the reconstructed density matrix.
    """
    density_matrices: ty.List[numpy.ndarray] = []
    # This reconstruction could potentially be performed in parallel.
    # Left as a TODO for the moment.
    for freqs in frequencies:
        # Build the projectors and the observed frequencies
        projectors: ty.List[numpy.ndarray] = list()
        observed_frequencies: ty.List[float] = list()
        for basis_change_name, state_projector in zip(
            basis.basis_change_circuit_names, basis.projector_states
        ):
            projectors.append(state_projector)
            observed_frequencies.append(freqs[basis_change_name].get(0, 0))
            projectors.append(get_orthogonal_state(state_projector))
            observed_frequencies.append(freqs[basis_change_name].get(1, 0))
        # Solving the system
        density_matrices.append(
            reconstruct_density_matrix(observed_frequencies, projectors)
        )
    return density_matrices


def post_process_tomography_results_grad(
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
    return frequencies_to_grad_reconstruction(frequencies, basis)
