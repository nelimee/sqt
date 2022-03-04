import numpy
from sqt import _constants


def couter(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
    """Complex conjugate outer product."""
    # TODO: was numpy.outer(a.T.conj(), b), might be a mistake
    return numpy.outer(a, b.T.conj())


def bloch_vector_to_density_matrix(s: numpy.ndarray) -> numpy.ndarray:
    """
    Take a 3-dimensional Bloch vector and returns the corresponding density matrix.

    :param s: A 3-dimensional real vector representing a point within the Bloch
        sphere.
    :return: a 2 by 2 density matrix corresponding to the given state.
    """
    return (
        _constants.I + s[0] * _constants.X + s[1] * _constants.Y + s[2] * _constants.Z
    ) / 2


def get_orthogonal_state(state: numpy.ndarray) -> numpy.ndarray:
    """Compute the only orthogonal quantum state of the given 1-qubit state."""
    projector_matrix: numpy.ndarray = numpy.outer(state.T.conj(), state)
    inverse_projector_matrix: numpy.ndarray = _constants.I - projector_matrix
    ra = numpy.sqrt(inverse_projector_matrix[0, 0])
    rb = numpy.sqrt(inverse_projector_matrix[1, 1])
    thetaa, thetab = -numpy.angle(state[1]), -numpy.angle(state[1]) + numpy.angle(
        inverse_projector_matrix[0, 1]
    )
    orthogonal_state: numpy.ndarray = numpy.array(
        [ra * numpy.exp(1.0j * thetaa), rb * numpy.exp(1.0j * thetab)],
        dtype=complex,
    )
    return orthogonal_state
