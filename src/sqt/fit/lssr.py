import numpy
import scipy.linalg
from qiskit import QuantumCircuit
from qiskit.result import Result

from sqt.basis.base import BaseMeasurementBasis
from sqt.counts import Counts
from sqt.fit._helpers import compute_frequencies


def _make_positive_semidefinite(
    mat: numpy.ndarray, epsilon: float = 0
) -> numpy.ndarray:
    """Rescale a Hermitian matrix to nearest postive semidefinite matrix.

    From https://github.com/Qiskit/qiskit-ignis/blob/81955597ebabe1870088e4adc606f7ad2d71ffa8/qiskit/ignis/verification/tomography/fitters/lstsq_fit.py#L124

    References:
        [1] J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502
            (2012). Open access: arXiv:1106.5458 [quant-ph].

    :note: this function looks very much like the projection operation in the grad.py
        file. It might be factored out in the future.

    Args:
        mat: a hermitian matrix.
        epsilon: (default: 0) the threshold for setting eigenvalues to
            zero. If epsilon > 0 positive eigenvalues below epsilon will
            also be set to zero.

    Raises:
        ValueError: If epsilon is negative

    Returns:
        The input matrix rescaled to have non-negative eigenvalues.
    """
    if epsilon is not None and epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    # Get the eigenvalues and eigenvectors of rho
    # eigenvalues are sorted in increasing order
    # v[i] <= v[i+1]
    dim = len(mat)
    v, w = scipy.linalg.eigh(mat)
    for j in range(dim):
        if v[j] < epsilon:
            tmp = v[j]
            v[j] = 0.0
            # Rescale remaining eigenvalues
            x = 0.0
            for k in range(j + 1, dim):
                x += tmp / (dim - (j + 1))
                v[k] = v[k] + tmp / (dim - (j + 1))

    # Build positive matrix from the rescaled eigenvalues
    # and the original eigenvectors
    mat_psd = numpy.zeros([dim, dim], dtype=complex)
    for j in range(dim):
        mat_psd += v[j] * numpy.outer(w[:, j], numpy.conj(w[:, j]))

    return mat_psd


def frequencies_to_lssr_reconstruction(
    frequencies: list[dict[str, Counts]],
    basis: BaseMeasurementBasis,
    epsilon: float = 1e-3,
    verbose: bool = False,
) -> list[numpy.ndarray]:
    """Compute the density matrix from the given frenquencies.

    This function constructs an observable matrix A that contains all the
    projectors of the observables used in the circuits submitted to the QPU.
    Note that the projectors, that should be 2 by 2 complex matrices, are
    flattened to form a 4-element row vector in order to build the matrix A.
    Then, the measurement results are gathered into a probability vector p,
    following the projectors ordering in A (i.e. the i-th entry in p
    corresponds to the empirical probability of observing the state from the
    projector in the i-th row of A).

    Finally the density matrix rho is computed as the solution to the linear
    system

    A rho = p

    This method does not ensure that rho is a density matrix as it might, due
    to imprecisions in p which might come from the noisy hardware, end up being
    non-physical. This is something TODO.

    Args:
        frenquencies: the estimated frequencies as a list of mappings
            {basis_change_str -> {state -> frequency}} where
            basis_change_str is the name of the quantum circuit
            performing the basis change, state is either "0" or "1" for
            1-qubit and frequency is the estimated frequency.
        basis: the tomography basis used.
        epsilon: a small float that is used to warn if the projection
            used for LSSR method changes too much the matrix. Basically,
            for a computed and non-necessarily semi-definite positive
            matrix rho, if || rho - proj(rho) || > epsilon where proj(.)
            is a projector to the space of semi-definite positive
            matrices, then a warning will be issued.
        verbose: if True, print warnings and information about the
            optimisation.

    Returns:
        the reconstructed density matrix.
    """
    density_matrices: list[numpy.ndarray] = []
    # This reconstruction could potentially be performed in parallel.
    # Left as a TODO for the moment.
    for freqs in frequencies:
        # Build the projectors and the observed frequencies
        A_rows: list[numpy.ndarray] = list()
        b_entries: list[float] = list()
        for basis_change_name, (state_projector, orthogonal_projector) in zip(
            basis.basis_change_circuit_names, basis.projectors
        ):
            A_rows.append(state_projector.ravel())
            b_entries.append(freqs[basis_change_name].get(0, 0))
            A_rows.append(orthogonal_projector.ravel())
            b_entries.append(freqs[basis_change_name].get(1, 0))
        A = numpy.array(A_rows)
        b = numpy.array(b_entries)
        # Solving the system
        rho_vec, *_ = scipy.linalg.lstsq(A, b)  # type: ignore
        # Warning: the solution obtained rho_vec is the vector representing
        #          the result density matrix but it needs to be transposed in
        #          order to be correct (do the maths and check this if you want).
        density_matrix = rho_vec.reshape((2, 2)).T
        # Project the density matrix to the closest SDP matrix.
        sdp_density_matrix = _make_positive_semidefinite(density_matrix)
        # Issue a warning if the projection changed significantly the density matrix.
        sdp_error = numpy.linalg.norm(density_matrix - sdp_density_matrix)
        if verbose and sdp_error > epsilon:
            print(
                "Warning: the returned density matrix has been changed "
                "significantly to be SDP: ||original - sdp||_2 = "
                f"{sdp_error:.2e}"
            )
        density_matrices.append(sdp_density_matrix)
    return density_matrices


def post_process_tomography_results_lssr(
    result: Result,
    tomographied_circuit: QuantumCircuit,
    basis: BaseMeasurementBasis,
    qubit_number: int = 1,
) -> list[numpy.ndarray]:
    """Compute and return the density matrix computed via state tomography.

    This function constructs an observable matrix A that contains all the
    projectors of the observables used in the circuits submitted to the QPU.
    Note that the projectors, that should be 2 by 2 complex matrices, are
    flattened to form a 4-element row vector in order to build the matrix A.
    Then, the measurement results are gathered into a probability vector p,
    following the projectors ordering in A (i.e. the i-th entry in p
    corresponds to the empirical probability of observing the state from the
    projector in the i-th row of A).

    Finally the density matrix rho is computed as the solution to the linear
    system

    A rho = p

    This method does not ensure that rho is a density matrix as it might, due
    to imprecisions in p which might come from the noisy hardware, end up being
    non-physical. This is something TODO.

    Args:
        result: the Result instance returned by the QPU after executing
            all the circuits returned by the
            one_qubit_tomography_circuits function.
        tomographied_circuit: the quantum circuit instance that is
            currently tomographied. Used to recover the circuit name.
        basis: the basis in which the measurements will be done.
        qubit_number: the number of qubits the parallel 1-qubit
            tomography should be performed on. Default to 1, i.e. no
            parallel execution.

    Returns:
        the 2 by 2 density matrix representing the prepared quantum
        state.
    """
    # Compute the frequencies
    frequencies: list[dict[str, Counts]] = compute_frequencies(
        result, tomographied_circuit, basis, qubit_number
    )

    return frequencies_to_lssr_reconstruction(frequencies, basis)
