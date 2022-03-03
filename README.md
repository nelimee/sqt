# Single Qubit Tomography

This project has been made during the Quantum Computing Summer School 2021 at Los Alamos National Laboratory with the help of:

- Carleton Coffrin
- Marc Vuffray
- Andrey Lokhov
- Jon Nelson

## Quick start

### Install

In order to start using the code in this repository, first install the `sqt` package with the command
```sh
python -m pip install -e .
```
The above command should be issued **in the main folder** of this repository.

### Using `sqt`

The `sqt` package can be used to implement quantum state tomography. It has been specifically designed for efficient 1-qubit tomography but might be improved in the future for efficient multi-qubit tomography. Moreover, `sqt` only supports [Qiskit](https://qiskit.org) for the moment. More frameworks might come if there is a need.

Using `sqt` is quite simple and the different steps are explained below.

#### 1. Pick your tomography basis

`sqt` implements 3 different 1-qubit tomography basis that you can choose from:

1. The `pauli` basis that measure along the X, Y and Z axis.
2. The `tetrahedral` basis that implements the SIC-POVM basis described [here](https://en.wikipedia.org/wiki/SIC-POVM#Simplest_example).
3. The `equidistant` basis that perform measurements in a user-defined number of projectors that are approximately equidistant from each other. This basis has not been used yet and might be useful when redundancy is desired.

```python
from sqt import EquidistantMeasurementBasis, PauliMeasurementBasis, TetrahedralMeasurementBasis

pauli_basis = PauliMeasurementBasis()
tetrahedral_basis = TetrahedralMeasurementBasis()
equidistant_basis = EquidistantMeasurementBasis(approximative_point_number=10)

# Define the basis that will be used in this README:
basis = tetrahedral_basis
```


#### 2. Construct the tomography circuits

Once the basis has been picked, a simple call to `one_qubit_tomography_circuits` will generate all the necessary quantum circuits to perform quantum tomography.

```python
from sqt import one_qubit_tomography_circuits

circuit = None  # Replace with a QuantumCircuit instance to tomography
tomography_circuits = one_qubit_tomography_circuits(
    circuit,         # The circuit to tomography
    basis,           # The basis used for tomography
    # Should we perform the tomography in parallel on several qubits? 
    # If is_parallel is True, qubit_number should be a positive integer.
    is_parallel=True,
    # Number of qubits the tomography circuits should be repeated on.
    qubit_number=7,
)
```

Parallel execution of 1-qubit tomography circuits is natively supported by `sqt` and is provided as simple keywords 
