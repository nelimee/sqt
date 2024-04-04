import typing as ty

import numpy
from qiskit import QuantumCircuit
from qiskit.quantum_info.states import DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator

from sqt.basis.tetrahedral import TetrahedralMeasurementBasis
from sqt.circuits import one_qubit_tomography_circuits
from sqt.fit.grad import post_process_tomography_results_grad
from sqt.fit.lssr import post_process_tomography_results_lssr
from sqt.fit.mle import post_process_tomography_results_mle

METHODS = {
    "grad": post_process_tomography_results_grad,
    "mle": post_process_tomography_results_mle,
    "lssr": post_process_tomography_results_lssr,
}
simulator = AerSimulator(method="automatic")

raw_circuit = QuantumCircuit(1, 1, name="Ry(pi/8)")
raw_circuit.ry(numpy.pi / 8, 0)
raw_circuit.rz(numpy.pi / 3, 0)

qubit_number = 5
shot_number = 2**15

basis = TetrahedralMeasurementBasis()
# basis = PauliMeasurementBasis()
# basis = EquidistantMeasurementBasis(10)
tomography_circuits = one_qubit_tomography_circuits(
    raw_circuit, basis, qubit_number=qubit_number
)
print(
    f"Running {len(tomography_circuits)} circuits, each with "
    f"{shot_number} shots and {qubit_number} qubits. The 1-qubit output "
    "density matrices of each qubit will be computed."
)
result = simulator.run(tomography_circuits, shots=shot_number).result()

density_matrices: ty.Dict[str, ty.List[numpy.ndarray]] = {}
for method in METHODS:
    print(f"Post-processing with method {method}...")
    density_matrices[method] = METHODS[method](
        result, raw_circuit, basis, qubit_number=qubit_number
    )

# This method is automatically added to the QuantumCircuit class by qiskit.
raw_circuit.save_density_matrix()  # type: ignore
dm_result = simulator.run(raw_circuit).result()
exact_density_matrix: numpy.ndarray = dm_result.results[0].data.density_matrix


print(
    "Fidelity of the quantum states reconstructed from the 1-qubit tomography experiment "
    "on the perfect simulator:"
)
exact_state = DensityMatrix(exact_density_matrix)
for i in range(qubit_number):
    print(f"Qubit {i}")
    for m in METHODS:
        obtained_state = DensityMatrix(density_matrices[m][i])
        fidelity: float = state_fidelity(exact_state, obtained_state, validate=False)
        print(f"\tUsing {m:>5}: {1 - fidelity:.4e}")
