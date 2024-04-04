import numpy

from sqt._typing import ComplexMatrix

I: ComplexMatrix = numpy.eye(2, dtype=complex)  # noqa: E741
X: ComplexMatrix = numpy.array([[0, 1], [1, 0]], dtype=complex)
Y: ComplexMatrix = 1.0j * numpy.array([[0, -1], [1, 0]], dtype=complex)
Z: ComplexMatrix = numpy.array([[1, 0], [0, -1]], dtype=complex)
