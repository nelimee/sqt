import math

import numpy
import scipy.linalg
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass


def _mod_2pi(angle: float) -> float:
    """Return the given angle within the range [0, 2pi]."""
    if angle < 0:
        angle += math.ceil(-angle / (2 * math.pi)) * 2 * math.pi
    elif angle > 2 * math.pi:
        angle -= math.floor(angle / (2 * math.pi)) * 2 * math.pi
    return angle


class Optimize1qGateIntoRzSX(TransformationPass):
    """Optimise any 1-qubit gate chain into 5 gates Rz SX Rz SX Rz.

    This is a strictly inferior implementation of the
    qiskit.transpiler.passes.Optimize1qGatesDecomposition pass, but
    it is used here because this pass garantee that the decomposition
    will always have 5 gates.
    """

    def __init__(self):
        """Optimise any 1-qubit gate chain into 5 gates Rz SX Rz SX Rz."""
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the Optimize1qGateIntoRzSX pass on `dag`.

        :param dag: the DAG to be optimized.
        :return: the optimized DAG.
        """
        runs = dag.collect_1q_runs()
        for r in runs:
            operator = r[0].op.to_matrix().copy()
            for gate in r[1:]:
                operator = gate.op.to_matrix().dot(operator)
            # Make the computed operator in SU2
            coeff = scipy.linalg.det(operator) ** (-0.5)
            phase = float(-numpy.angle(coeff))
            operator *= coeff
            # Computing Euler angles
            theta: float = 2 * math.atan2(abs(operator[1, 0]), abs(operator[0, 0]))
            phi: float = numpy.angle(operator[1, 1]) + numpy.angle(operator[1, 0])
            lambd: float = numpy.angle(operator[1, 1]) - numpy.angle(operator[1, 0])

            # Creating the DAGCircuit instance that will replace all the
            # 1-qubit operations
            replacement_circ = QuantumCircuit(1)
            replacement_circ.rz(_mod_2pi(lambd), 0)
            replacement_circ.sx(0)
            replacement_circ.rz(_mod_2pi(theta + numpy.pi), 0)
            replacement_circ.sx(0)
            replacement_circ.rz(_mod_2pi(phi + numpy.pi), 0)
            replacement_circ.global_phase = _mod_2pi(numpy.pi / 2 - phase)
            replacement_dag: DAGCircuit = circuit_to_dag(replacement_circ)
            dag.substitute_node_with_dag(r[0], replacement_dag)
            # Delete the other nodes in the run
            for node in r[1:]:
                dag.remove_op_node(node)
        return dag


def compile_circuits(circuits: list[QuantumCircuit]) -> list[QuantumCircuit]:
    """Merge 1-qubit gates with the Optimize1qGateIntoRzSX pass."""
    from qiskit.transpiler import PassManager

    from sqt.passes import Optimize1qGateIntoRzSX

    pass_sspin = Optimize1qGateIntoRzSX()
    pm_sspin = PassManager(passes=[pass_sspin])
    result = pm_sspin.run(circuits)
    if isinstance(result, QuantumCircuit):
        # Should not be possible in theory, but makes typing checker happy
        return [result]
    else:
        return result
