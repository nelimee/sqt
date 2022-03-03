import typing as ty

import numpy
import scipy.linalg
from qiskit import BasicAer, QuantumCircuit, execute
from qiskit.result import Result

from sqt import _constants
from sqt.basis import BaseMeasurementBasis, PauliMeasurementBasis
from sqt.fit._helpers import _couter, compute_frequencies


def _make_positive_semidefinite(
    mat: numpy.ndarray, epsilon: ty.Optional[float] = 0
) -> numpy.ndarray:
    """
    Rescale a Hermitian matrix to nearest postive semidefinite matrix.

    From https://github.com/Qiskit/qiskit-ignis/blob/81955597ebabe1870088e4adc606f7ad2d71ffa8/qiskit/ignis/verification/tomography/fitters/lstsq_fit.py#L124

    References:
        [1] J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502
            (2012). Open access: arXiv:1106.5458 [quant-ph].

    :param mat: a hermitian matrix.
    :param epsilon: (default: 0) the threshold for setting eigenvalues to
        zero. If epsilon > 0 positive eigenvalues below epsilon will also
        be set to zero.
    :raise ValueError: If epsilon is negative
    :returns: The input matrix rescaled to have non-negative eigenvalues.
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
    frequencies: ty.Dict[str, ty.Dict[str, float]],
    basis: BaseMeasurementBasis,
    epsilon: float = 1e-3,
) -> numpy.ndarray:
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

    :param frenquencies:
    :param basis:
    :return:
    """
    # Build the "measurement matrix"
    simulator = BasicAer.get_backend("statevector_simulator")
    A_rows: ty.List[numpy.ndarray] = list()
    b_entries: ty.List[float] = list()
    for basis_change_circuit in basis.basis_change_circuits():
        inverted_basis_change_circuit = basis_change_circuit.inverse()
        state: numpy.ndarray = (
            execute(inverted_basis_change_circuit, simulator)
            .result()
            .get_statevector(inverted_basis_change_circuit.name)
        )
        projector = _couter(state, state)
        A_rows.append(projector.ravel())
        A_rows.append((_constants.I - projector).ravel())
        b_entries.append(frequencies[basis_change_circuit.name].get("0", 0))
        b_entries.append(frequencies[basis_change_circuit.name].get("1", 0))
    A = numpy.array(A_rows)
    b = numpy.array(b_entries)
    # Solving the system
    rho_vec, residues, rank, s = scipy.linalg.lstsq(A, b)
    density_matrix = rho_vec.reshape((2, 2))
    # Project the density matrix to the closest SDP matrix.
    sdp_density_matrix = _make_positive_semidefinite(density_matrix)
    # Issue a warning if the projection changed significantly the density matrix.
    sdp_error = numpy.linalg.norm(density_matrix - sdp_density_matrix)
    if sdp_error > epsilon:
        print(
            "Warning: the returned density matrix has been changed "
            "significantly to be SDP: ||original - sdp||_2 = "
            f"{sdp_error:.2e}"
        )

    return sdp_density_matrix


def post_process_tomography_results_lssr(
    result: Result,
    tomographied_circuit: QuantumCircuit,
    qubit_index: ty.Optional[int] = None,
    is_parallel: bool = False,
    basis: ty.Optional[BaseMeasurementBasis] = None,
) -> numpy.ndarray:
    """
    Compute and return the density matrix computed via state tomography.

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

    return frequencies_to_lssr_reconstruction(frequencies, basis)
