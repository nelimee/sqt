# Single Qubit Tomography

This project has been made during the Quantum Computing Summer School 2021 at Los Alamos National Laboratory with the help of:

- Carleton Coffrin
- Marc Vuffray
- Andrey Lokhov
- Jon Nelson

and continued during my PhD.

## Quick start

### Install

In order to start using the code in this repository, first install the `sqt` package with the command
```sh
git clone git@github.com/nelimee/sqt
python -m pip install -e sqt/
```

### Command line interface

The `sqt` package provides scripts that are installed along with the package.

#### `sqt_bloch_tomography_submit`

This script makes the circuit building and submission process easier by providing a simple interface to the user.

The script inputs can be listed with the `--help` option:
```
>>> sqt_bloch_tomography_submit --help
usage: sqt_bloch_tomography_submit [-h] [--equidistant-points EQUIDISTANT_POINTS] [--hub HUB] [--group GROUP] [--project PROJECT] [--backup-dir BACKUP_DIR] [--rep-delay REP_DELAY] [--shots SHOTS]
                                   [--delay-dt DELAY_DT] [--max-qubits MAX_QUBITS] [--local-backend] [--noisy-simulator]
                                   backend approximate_point_number {pauli,tetrahedral,equidistant}

Execute the circuits to perform state tomography for approximately uniformly distributed quantum states over the Bloch sphere.

positional arguments:
  backend               Backend to perform tomography on.
  approximate_point_number
                        Approximate number of points used to cover the Bloch sphere. The actual number of points used might vary a little bit from this value. Each point will be tomographied independently.
  {pauli,tetrahedral,equidistant}
                        Name of the tomography basis used to perform quantum state tomography.

optional arguments:
  -h, --help            show this help message and exit
  --equidistant-points EQUIDISTANT_POINTS
                        If basis is 'equidistant', the number of approximately equidistant projectors that will be used. Else, this option is ignored.
  --hub HUB             Hub of your IBMQ provider. Defaults to 'ibm-q' available to all users.
  --group GROUP         Group of your IBMQ provider. Defaults to 'open' available to all users.
  --project PROJECT     Project of your IBMQ provider. Defaults to 'main' available to all users.
  --backup-dir BACKUP_DIR
                        Directory used to save the data needed to post-process job results.
  --rep-delay REP_DELAY
                        Delay between each shot. Default to the backend default value.
  --shots SHOTS         Number of shots performed for each circuit.
  --delay-dt DELAY_DT   Duration (in dt) of the delay to insert before state tomography.
  --max-qubits MAX_QUBITS
                        Maximum number of qubits that should be used.
  --local-backend       If present, the backend used is a local one.
  --noisy-simulator     If present, the given IBMQ backend will be used to initialise a noisy simulator. Implies '--local-backend'.
```

Here are some examples of usage:
```sh
>>> sqt_bloch_tomography_submit --hub ibm-q --group open --project main ibmq_lima 100 tetrahedral
# In the following, the argument imbq_lima is ignored due to the option
# --local-backend being given.
>>> sqt_bloch_tomography_submit ibmq_lima 100 tetrahedral --local-backend
# In the following, a noisy simulator initialised with ibmq_lima current calibrations will
# be used.
>>> sqt_bloch_tomography_submit --hub ibm-q --group open --project main ibmq_lima 100 tetrahedral --noisy-simulator
```

You can of course use a custom provider, change the `rep_delay` value for backends that implement this feature, specify a maximum number of qubits to use (useful for the simulators and can be used to limit the number of qubits used on a given backend) or change the directory that will be used to store the backup file.

The backup file is the main output of this script: it contains all the needed information for `sqt_bloch_tomography_recover`.

#### `sqt_bloch_tomography_recover`

This script takes as input a backup file created with `sqt_bloch_tomography_submit` and reconstructs the density matrices representing the tomographied quantum state with the provided reconstruction method.

The script inputs can be listed with the `--help` option:
```
>>> sqt_bloch_tomography_recover --help
usage: sqt_bloch_tomography_recover [-h]
                                    backup_filepath {mle,pauli,lssr,grad}
                                    [{mle,pauli,lssr,grad} ...]

Post-process the job result and compute the reconstructed density matrices.

positional arguments:
  backup_filepath       Backup file path that has been saved during the job submission.
  {mle,pauli,lssr,grad}
                        Post-processing method used to reconstruct the density matrices.

optional arguments:
  -h, --help            show this help message and exit
```

Here are some examples of usage:
```sh
# Replace the "[file with .pkl extension]" with the backup file
# obtained with sqt_bloch_tomography_submit
>>> sqt_bloch_tomography_recover [file with .pkl extension] grad
>>> sqt_bloch_tomography_recover [file with .pkl extension] lssr grad mle
```

## Using the `sqt` package directly

The `sqt` package can be used to implement quantum state tomography. It has been specifically designed for efficient 1-qubit tomography but might be improved in the future for efficient multi-qubit tomography. Moreover, `sqt` only supports [Qiskit](https://qiskit.org) for the moment. More frameworks might come if there is a need.

Using `sqt` is quite simple and the different steps are explained below.

### 1. Pick your tomography basis

`sqt` implements 3 different 1-qubit tomography basis that you can choose from:

1. The `pauli` basis that measure along the X, Y and Z axis.
2. The `tetrahedral` basis that implements the SIC-POVM basis described [here](https://en.wikipedia.org/wiki/SIC-POVM#Simplest_example).
3. The `equidistant` basis that perform measurements in a user-defined number of projectors that are approximately equidistant from each other. This basis has not been used yet and might be useful when redundancy is desired.

```python
from sqt.basis.equidistant import EquidistantMeasurementBasis
from sqt.basis.pauli import PauliMeasurementBasis
from sqt.basis.tetrahedral import TetrahedralMeasurementBasis

pauli_basis = PauliMeasurementBasis()
tetrahedral_basis = TetrahedralMeasurementBasis()
equidistant_basis = EquidistantMeasurementBasis(approximative_point_number=10)

# Define the basis that will be used in this README:
basis = tetrahedral_basis
```


### 2. Construct the tomography circuits

Once the basis has been picked, a simple call to `one_qubit_tomography_circuits` will generate all the necessary quantum circuits to perform quantum tomography.

```python
from sqt.circuits import one_qubit_tomography_circuits

circuit = None  # Replace with a QuantumCircuit instance to tomography
                # No measurements should be appended to this circuit!
tomography_circuits = one_qubit_tomography_circuits(
    circuit,         # The circuit to tomography
    basis,           # The basis used for tomography
    # Number of qubits the tomography circuits should be repeated on.
    qubit_number=7,
)
```

Parallel execution of 1-qubit tomography circuits is natively supported by `sqt` and is provided as simple keywords 

### 3. Post process the results

Now that the quantum circuits returned by `one_qubit_tomography_circuits` have been executed, they should be processed in order to recover the density matrix.
```python
from qiskit import execute

from sqt.fit.grad import post_process_tomography_results_grad
from sqt.fit.mle import post_process_tomography_results_mle

backend = None # Fill this with the backend of your choice
result = execute(tomography_circuits, backend).result()

density_matrices = post_process_tomography_results_mle(
    result, 
    # Warning: this is the original circuit without basis change nor
    #          measurements at the end. Do not change it between the
    #          call to one_qubit_tomography_circuits and here!
    circuit, 
    basis,
    qubit_number=7
)
```
