import typing as ty
import numpy

from qiskit import QuantumCircuit
from qiskit.quantum_info.states import DensityMatrix, state_fidelity
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator


from sqt.circuits import one_qubit_tomography_circuits
from sqt.basis.tetrahedral import TetrahedralMeasurementBasis
from sqt.fit.grad import post_process_tomography_results_grad
from sqt.fit.mle import post_process_tomography_results_mle
from sqt.fit.lssr import post_process_tomography_results_lssr

hub = "ibm-q-lanl"
group = "lanl"
project = "quantum-simulati"
backend_name = "ibm_algiers"

print(f"Recovering IBMQ backend {backend_name} with {hub}/{group}/{project}.")
service = QiskitRuntimeService(
    channel="ibm_quantum", instance=f"{hub}/{group}/{project}"
)
if not service.active_account():
    raise RuntimeError(f"Could not load account with '{hub}' '{group}' '{project}'.")
backend = service.get_backend("ibm_algiers")
print(f"Initialising a noisy simulator with {backend_name} calibrations.")
simulator = AerSimulator.from_backend(backend)
qubit_number: int = 5
max_shots: int = simulator.configuration().max_shots

METHODS = {
    "grad": post_process_tomography_results_grad,
    "mle": post_process_tomography_results_mle,
    "lssr": post_process_tomography_results_lssr,
}
exact_simulator = AerSimulator(method="automatic")

raw_circuit = QuantumCircuit(1, 1, name="Ry(pi/8)")
raw_circuit.ry(numpy.pi / 8, 0)
raw_circuit.rz(numpy.pi / 3, 0)

basis = TetrahedralMeasurementBasis()
tomography_circuits = one_qubit_tomography_circuits(
    raw_circuit, basis, qubit_number=qubit_number
)

print(
    f"Running {len(tomography_circuits)} circuits, each with "
    f"{max_shots} shots and {qubit_number} qubits. The 1-qubit output "
    "density matrices of each qubit will be computed."
)
result = simulator.run(tomography_circuits, shots=max_shots).result()

density_matrices: ty.Dict[str, ty.List[numpy.ndarray]] = {}
for method in METHODS:
    density_matrices[method] = METHODS[method](
        result, raw_circuit, basis, qubit_number=qubit_number
    )

# This method
raw_circuit.save_density_matrix()  # type: ignore
dm_result = exact_simulator.run(raw_circuit).result()
exact_density_matrix: numpy.ndarray = dm_result.results[0].data.density_matrix


print(
    "Fidelity of the quantum states reconstructed from the 1-qubit tomography experiment "
    f"on the noisy simulator initialised with {backend_name} calibrations:"
)

exact_state = DensityMatrix(exact_density_matrix)
# print(exact_state)
for i in range(qubit_number):
    print(f"Qubit {i}")
    for m in METHODS:
        obtained_state = DensityMatrix(density_matrices[m][i])
        # print(obtained_state)
        fidelity: float = state_fidelity(exact_state, obtained_state, validate=False)
        print(f"\tUsing {m:>5}: {1 - fidelity:.4e}")
# print(exact_state)

# print(raw_circuit.draw())
# for t in tomography_circuits:
#     print(t.draw())
