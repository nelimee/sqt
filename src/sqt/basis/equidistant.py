from qiskit import QuantumCircuit

import numpy
import numpy.typing as npt


from sqt.basis.base import BaseMeasurementBasis


def get_equidistant_points(N: int) -> list[tuple[float, float, float]]:
    """Generate approximately n points evenly distributed accros the 3-d sphere.

    This function tries to find approximately n points (might be a little less
    or more) that are evenly distributed accros the 3-dimensional unit sphere.

    The algorithm used is described in
    https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf.
    """
    # Unit sphere
    r = 1

    points: list[tuple[float, float, float]] = list()

    a = 4 * numpy.pi * r**2 / N
    d = numpy.sqrt(a)
    m_v = int(numpy.round(numpy.pi / d))
    d_v = numpy.pi / m_v
    d_phi = a / d_v

    for m in range(m_v):
        v = numpy.pi * (m + 0.5) / m_v
        # m_phi = int(numpy.round(2 * numpy.pi * numpy.sin(v / d_phi)))
        m_phi = int(numpy.round(2 * numpy.pi * numpy.sin(v) / d_phi))
        for n in range(m_phi):
            phi = 2 * numpy.pi * n / m_phi
            points.append(
                (
                    numpy.sin(v) * numpy.cos(phi),
                    numpy.sin(v) * numpy.sin(phi),
                    numpy.cos(v),
                )
            )
    return points


def points_to_xyz(
    points: list[tuple[float, float, float]],
) -> tuple[
    npt.NDArray[numpy.float_], npt.NDArray[numpy.float_], npt.NDArray[numpy.float_]
]:
    """Transform a list of 3-dimensional points into 3 lists of coordinates.

    :param points: a list of 3-dimensional points like
        [[x1, y1, z1], [x2, y2, z2]].
    :return: 3 arrays of coordinates like ([x1, x2], [y1, y2], [z1, z2]).
    """
    p = numpy.asarray(points)
    return p[:, 0], p[:, 1], p[:, 2]


def point_to_circuit(point: tuple[float, float, float], name: str) -> QuantumCircuit:
    """Transform a pure state into a QuantumCircuit that prepares this state.

    :param points: a pure state given as a 3-dimensional point in the cartesian
        coordinate system.
    :return: a QuantumCircuit instance that prepare the state given by the
        point parameter if the initial state is |0>.
    """
    circuit = QuantumCircuit(1, name=name)
    theta = numpy.arccos(point[2])
    phi = numpy.angle(point[0] + 1.0j * point[1])
    circuit.ry(theta, 0)
    circuit.rz(phi, 0)
    return circuit


def get_approximately_equidistant_circuits(
    approximate_point_number: int,
) -> list[QuantumCircuit]:
    """Construct and returns circuits that are approximately equidistant.

    This function will construct approximately approximate_point_number
    quantum circuits that will prepare quantum states that are approximately
    equidistant when placed on the Bloch sphere.

    :param approximate_point_number: number of circuits that will be generated.
        This is only an approximation, the actual number of circuits that will
        be generated might be slightly lower or higher.
    :return: the generated circuits. Each circuit will have a name that is
        the list representation of the point it prepares. This point can
        be retrieved with the following code (yes it is ugly, I am searching
        for a better alternative):
            circuits = get_approximately_equidistant_circuits(10)
            points = [eval(c.name) for c in circuits]
    """
    return [
        point_to_circuit(point, f"{point}")
        for point in get_equidistant_points(approximate_point_number)
    ]


class EquidistantMeasurementBasis(BaseMeasurementBasis):
    """A basis composed of approximately equidistant states on the Bloch sphere.

    This basis is composed of a variable number of measurements (given by the
    user when creating the class instance) that project the quantum state
    onto quantum states that are approximately equidistant on the Bloch sphere.

    See the get_equidistant_points function docstring to understand the
    algorithm used to create approximately equidistant points on the unit
    (Bloch) sphere.
    """

    def __init__(self, approximative_point_number: int):
        """Initialise an instance of EquidistantMeasurementBasis.

        :param approximative_point_number: the number of points requested by
            the user. Due to the algorithm used to generate approximately
            equidistant points, the actual number of circuits in the basis
            might not be exactly the number requested by the user.
        """
        super().__init__(f"equidistant-{approximative_point_number}")
        self._approximative_point_number: int = approximative_point_number
        self._basis_change_circuits: list[QuantumCircuit] = list()
        for i, point in enumerate(
            get_equidistant_points(self._approximative_point_number)
        ):
            self._basis_change_circuits.append(
                point_to_circuit(point, str(i)).inverse()
            )

    @property
    def basis_change_circuits(self) -> list[QuantumCircuit]:
        """Return approximately n basis changes that are approximately equidistant.

        This function tries to find approximately n points (might be a little
        less or more) that are evenly distributed accros the 3-dimensional unit
        sphere and generates the quantum circuits needed to perform a
        measurement in the basis represented by each of these points.

        :return: all the basis change needed to perform the state tomography
            process.
        """
        return self._basis_change_circuits

    @property
    def size(self) -> int:
        return len(self._basis_change_circuits)
